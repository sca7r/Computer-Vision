"""api.py — FastAPI server: MJPEG stream + live dashboard + REST endpoints."""

import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from state import state

app = FastAPI(title=" Monitoring System", docs_url="/docs")


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HTML  —  Industrial dark theme, monospace, amber accents
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title> Monitoring System</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #080b0f;
    --surface:  #0e1318;
    --border:   #1e2830;
    --amber:    #f5a623;
    --green:    #22d97a;
    --red:      #ff3b5c;
    --blue:     #38bdf8;
    --muted:    #4a5a6a;
    --text:     #c8d8e8;
    --mono:     'Share Tech Mono', monospace;
    --sans:     'Barlow', sans-serif;
  }

  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    overflow-x: hidden;
  }

  /* ── Grid layout ── */
  .layout {
    display: grid;
    grid-template-rows: 56px 1fr auto;
    grid-template-columns: 1fr 340px;
    grid-template-areas:
      "header  header"
      "video   sidebar"
      "alerts  alerts";
    min-height: 100vh;
    gap: 1px;
    background: var(--border);
  }

  /* ── Header ── */
  header {
    grid-area: header;
    background: var(--surface);
    display: flex;
    align-items: center;
    padding: 0 24px;
    gap: 16px;
    border-bottom: 1px solid var(--border);
  }

  .logo-mark {
    width: 28px; height: 28px;
    border: 2px solid var(--amber);
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--amber);
    letter-spacing: 1px;
    flex-shrink: 0;
  }

  header h1 {
    font-family: var(--mono);
    font-size: 13px;
    letter-spacing: 3px;
    color: var(--text);
    text-transform: uppercase;
  }

  .status-pill {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--muted);
  }

  .pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
  }

  /* ── Video panel ── */
  .video-panel {
    grid-area: video;
    background: var(--bg);
    display: flex;
    flex-direction: column;
  }

  .panel-label {
    padding: 10px 16px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 3px;
    color: var(--muted);
    text-transform: uppercase;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .panel-label::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--amber);
  }

  #stream-img {
    width: 100%;
    height: calc(100% - 37px);
    object-fit: contain;
    display: block;
    background: #000;
  }

  /* ── Sidebar ── */
  .sidebar {
    grid-area: sidebar;
    background: var(--surface);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }

  /* ── Stat cards ── */
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
  }

  .stat-card {
    background: var(--surface);
    padding: 18px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .stat-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 3px;
    color: var(--muted);
    text-transform: uppercase;
  }

  .stat-value {
    font-family: var(--mono);
    font-size: 28px;
    font-weight: 400;
    color: var(--text);
    line-height: 1;
  }

  .stat-value.amber { color: var(--amber); }
  .stat-value.green { color: var(--green); }
  .stat-value.red   { color: var(--red);   }

  /* ── Face table ── */
  .face-table-wrap {
    padding: 0;
    flex: 1;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 12px;
  }

  thead tr {
    background: var(--bg);
  }

  th {
    padding: 8px 12px;
    text-align: left;
    font-size: 9px;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    font-weight: 400;
    border-bottom: 1px solid var(--border);
  }

  td {
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }

  tr:last-child td { border-bottom: none; }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: 600;
  }

  .badge-awake   { background: rgba(34,217,122,.15); color: var(--green); border: 1px solid rgba(34,217,122,.3); }
  .badge-sleeping{ background: rgba(255,59,92,.15);  color: var(--red);   border: 1px solid rgba(255,59,92,.3);  animation: flash 0.8s ease-in-out infinite; }

  @keyframes flash {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
  }

  .no-faces {
    padding: 32px 16px;
    text-align: center;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 2px;
  }

  /* ── Alert log ── */
  .alerts-panel {
    grid-area: alerts;
    background: var(--surface);
    border-top: 1px solid var(--border);
    max-height: 200px;
    overflow-y: auto;
  }

  .alert-row {
    display: grid;
    grid-template-columns: 160px 60px 1fr;
    padding: 7px 16px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    gap: 16px;
    align-items: center;
  }

  .alert-row:last-child { border-bottom: none; }

  .alert-ts   { color: var(--muted); }
  .alert-face { color: var(--amber); }

  .alert-kind-SLEEPING { color: var(--red); }
  .alert-kind-EMAIL    { color: var(--blue); }
  .alert-kind-SMS      { color: var(--amber); }

  .no-alerts {
    padding: 24px 16px;
    text-align: center;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 2px;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>
<div class="layout">

  <!-- HEADER -->
  <header>
    <div class="logo-mark">MS</div>
    <h1> Monitoring System</h1>
    <div class="status-pill">
      <div class="pulse-dot" id="live-dot"></div>
      <span id="live-label">LIVE</span>
    </div>
  </header>

  <!-- VIDEO -->
  <div class="video-panel">
    <div class="panel-label">Camera Feed</div>
    <img id="stream-img" src="/stream" alt="Camera stream">
  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">

    <div class="panel-label">System Stats</div>

    <div class="stat-grid">
      <div class="stat-card">
        <span class="stat-label">FPS</span>
        <span class="stat-value amber" id="stat-fps">–</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Faces</span>
        <span class="stat-value" id="stat-faces">–</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Sleeping</span>
        <span class="stat-value red" id="stat-sleeping">–</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Alerts</span>
        <span class="stat-value amber" id="stat-alerts">–</span>
      </div>
    </div>

    <div class="panel-label" style="margin-top:1px">Per-Face Status</div>

    <div class="face-table-wrap">
      <div id="face-table-body">
        <div class="no-faces">NO FACES DETECTED</div>
      </div>
    </div>

  </div>

  <!-- ALERTS LOG -->
  <div class="alerts-panel">
    <div class="panel-label">Alert Log</div>
    <div id="alert-log">
      <div class="no-alerts">NO ALERTS YET</div>
    </div>
  </div>

</div>

<script>
  const $ = id => document.getElementById(id);

  async function refreshStats() {
    try {
      const r = await fetch('/stats');
      const d = await r.json();

      $('stat-fps').textContent     = d.fps;
      $('stat-faces').textContent   = d.face_count;
      $('stat-sleeping').textContent = d.sleeping_count;

      // Per-face table
      if (d.faces.length === 0) {
        $('face-table-body').innerHTML = '<div class="no-faces">NO FACES DETECTED</div>';
      } else {
        const rows = d.faces.map(f => `
          <tr>
            <td>#${f.index + 1}</td>
            <td>${f.ear.toFixed(3)}</td>
            <td><span class="badge badge-${f.status.toLowerCase()}">${f.status}</span></td>
            <td>${f.blinks}</td>
          </tr>`).join('');
        $('face-table-body').innerHTML = `
          <table>
            <thead><tr><th>Face</th><th>EAR</th><th>Status</th><th>Blinks</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>`;
      }

      // Pulse dot colour
      $('live-dot').style.background = d.sleeping_count > 0 ? 'var(--red)' : 'var(--green)';
      $('live-dot').style.boxShadow  = d.sleeping_count > 0
        ? '0 0 8px var(--red)' : '0 0 6px var(--green)';

    } catch (_) { /* stream may be momentarily unavailable */ }
  }

  async function refreshAlerts() {
    try {
      const r = await fetch('/alerts');
      const alerts = await r.json();
      $('stat-alerts').textContent = alerts.length;

      if (alerts.length === 0) {
        $('alert-log').innerHTML = '<div class="no-alerts">NO ALERTS YET</div>';
        return;
      }
      $('alert-log').innerHTML = alerts.slice(0, 50).map(a => `
        <div class="alert-row">
          <span class="alert-ts">${a.timestamp}</span>
          <span class="alert-face">FACE #${a.face_index + 1}</span>
          <span class="alert-kind-${a.kind}">${a.kind}</span>
        </div>`).join('');
    } catch (_) {}
  }

  // Poll every 500 ms for stats, 2s for alerts
  setInterval(refreshStats,  500);
  setInterval(refreshAlerts, 2000);
  refreshStats();
  refreshAlerts();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/stream")
async def video_stream():
    """MJPEG stream — consumed by the <img> tag in the dashboard."""
    def generate():
        while state.running:
            jpg = state.get_frame()
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            time.sleep(0.033)   # ~30 fps max to browser

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stats")
async def get_stats():
    return JSONResponse(state.get_stats())


@app.get("/alerts")
async def get_alerts():
    return JSONResponse(state.get_alerts())
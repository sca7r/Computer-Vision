"""alerts.py — Email and SMS alerters that fire in background threads."""

import smtplib
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import EmailConfig, SMSConfig


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════════════════════════════════════

class EmailAlerter:

    def __init__(self, cfg: "EmailConfig"):
        self.cfg       = cfg
        self._last     = 0.0
        self._lock     = threading.Lock()

    def trigger(self, face_index: int) -> bool:
        """
        Send an alert email if cooldown has elapsed.
        Returns True if a send was dispatched, False if suppressed.
        """
        now = time.time()
        with self._lock:
            if now - self._last < self.cfg.cooldown:
                return False
            self._last = now

        threading.Thread(
            target=self._send, args=(face_index,), daemon=True
        ).start()
        return True

    def _send(self, face_index: int):
        try:
            ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = MIMEMultipart("alternative")
            msg["From"]    = self.cfg.sender
            msg["To"]      = ", ".join(self.cfg.recipients)
            msg["Subject"] = f"  Alert — Sleeping Detected  [{ts}]"

            plain = (
                f" Monitoring System\n"
                f"{'─' * 40}\n"
                f"Time   : {ts}\n"
                f"Face   : #{face_index + 1}\n"
                f"Status : SLEEPING detected\n\n"
                f"Please check the person immediately.\n"
            )
            html = f"""
<html><body style="font-family:monospace;background:#0a0a0f;color:#e0e0e0;padding:24px">
  <h2 style="color:#ff4444"> SLEEPING DETECTED</h2>
  <table style="border-collapse:collapse">
    <tr><td style="padding:4px 16px 4px 0;color:#888">Time</td>
        <td style="color:#fff">{ts}</td></tr>
    <tr><td style="padding:4px 16px 4px 0;color:#888">Face</td>
        <td style="color:#fff">#{face_index + 1}</td></tr>
    <tr><td style="padding:4px 16px 4px 0;color:#888">Status</td>
        <td style="color:#ff4444;font-weight:bold">SLEEPING</td></tr>
  </table>
  <p style="margin-top:24px;color:#888">Sent by  Monitoring System</p>
</body></html>
"""
            msg.attach(MIMEText(plain, "plain"))
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port) as srv:
                srv.ehlo()
                srv.starttls()
                srv.login(self.cfg.sender, self.cfg.password)
                srv.sendmail(self.cfg.sender, self.cfg.recipients, msg.as_string())

            print(f"[ALERT] Email sent — face #{face_index + 1}")

        except Exception as exc:
            print(f"[ERROR] Email failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# SMS  (Twilio)
# ══════════════════════════════════════════════════════════════════════════════

class SMSAlerter:

    def __init__(self, cfg: "SMSConfig"):
        self.cfg   = cfg
        self._last = 0.0
        self._lock = threading.Lock()
        self._client = None

        try:
            from twilio.rest import Client
            self._client = Client(cfg.twilio_sid, cfg.twilio_token)
        except ImportError:
            print("[WARNING] twilio not installed — SMS disabled.  pip install twilio")
        except Exception as exc:
            print(f"[WARNING] Twilio init failed: {exc}")

    def trigger(self, face_index: int) -> bool:
        if not self._client:
            return False
        now = time.time()
        with self._lock:
            if now - self._last < self.cfg.cooldown:
                return False
            self._last = now

        threading.Thread(
            target=self._send, args=(face_index,), daemon=True
        ).start()
        return True

    def _send(self, face_index: int):
        ts   = datetime.now().strftime("%H:%M:%S")
        body = (
            f" ALERT\n"
            f"Face #{face_index + 1} SLEEPING\n"
            f"Time: {ts}\n"
            f"Check  immediately."
        )
        try:
            for number in self.cfg.to_numbers:
                self._client.messages.create(
                    body  = body,
                    from_ = self.cfg.from_number,
                    to    = number,
                )
            print(f"[ALERT] SMS sent — face #{face_index + 1}")
        except Exception as exc:
            print(f"[ERROR] SMS failed: {exc}")
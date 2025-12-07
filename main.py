import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles


def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR)"""
    # Vertical distances
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear


def run_hand_and_face_tracking_with_sleep_detection():
    # EAR threshold and consecutive frames
    EAR_THRESHOLD = 0.25  # Below this = eyes closed
    CONSEC_FRAMES = 20    # Eyes closed for this many frames = sleeping
    
    # Counters
    frame_counter = 0
    blink_counter = 0
    status = "AWAKE"
    
    # MediaPipe Face Mesh eye landmark indices
    # Right eye landmarks (6 points)
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    # Left eye landmarks (6 points)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    print("Camera opened successfully!")

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        
        print("MediaPipe initialized. Press 'q' to quit.")
        
        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb)

            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Process face and detect sleep/awake
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Draw face contours
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    
                    # Extract eye landmarks
                    landmarks = face_landmarks.landmark
                    h, w, _ = frame.shape
                    
                    # Get right eye coordinates
                    right_eye = []
                    for idx in RIGHT_EYE:
                        x = int(landmarks[idx].x * w)
                        y = int(landmarks[idx].y * h)
                        right_eye.append((x, y))
                    
                    # Get left eye coordinates
                    left_eye = []
                    for idx in LEFT_EYE:
                        x = int(landmarks[idx].x * w)
                        y = int(landmarks[idx].y * h)
                        left_eye.append((x, y))
                    
                    # Calculate EAR for both eyes
                    right_ear = calculate_ear(right_eye)
                    left_ear = calculate_ear(left_eye)
                    avg_ear = (right_ear + left_ear) / 2.0
                    
                    # Check if eyes are closed
                    if avg_ear < EAR_THRESHOLD:
                        frame_counter += 1
                        
                        if frame_counter >= CONSEC_FRAMES:
                            status = "SLEEPING"
                            # Draw red warning
                            cv2.putText(frame, "ALERT: SLEEPING!", (10, 80),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    else:
                        if frame_counter >= CONSEC_FRAMES:
                            blink_counter += 1
                        
                        frame_counter = 0
                        status = "AWAKE"
                    
                    # Display EAR value
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display status
                    color = (0, 255, 0) if status == "AWAKE" else (0, 0, 255)
                    cv2.putText(frame, f"Status: {status}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Display blink counter
                    cv2.putText(frame, f"Blinks: {blink_counter}", (10, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand and Face Tracking with Sleep Detection", cv2.flip(frame, 1))
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_hand_and_face_tracking_with_sleep_detection()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

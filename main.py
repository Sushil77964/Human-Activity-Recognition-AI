import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utility
mp_draw = mp.solutions.drawing_utils

previous_left_ankle_y = None
# ---------------- Activity Detection ----------------
def detect_activity(landmarks):
    global previous_left_ankle_y

    LEFT_SHOULDER = landmarks[11]
    RIGHT_SHOULDER = landmarks[12]

    LEFT_WRIST = landmarks[15]
    RIGHT_WRIST = landmarks[16]

    LEFT_HIP = landmarks[23]
    RIGHT_HIP = landmarks[24]

    LEFT_KNEE = landmarks[25]
    RIGHT_KNEE = landmarks[26]

    LEFT_ANKLE = landmarks[27]

    activity = "Standing"

    # Hand Raised
    if LEFT_WRIST.y < LEFT_SHOULDER.y or RIGHT_WRIST.y < RIGHT_SHOULDER.y:
        activity = "Hand Raised"

    # Sitting
    elif LEFT_HIP.y > LEFT_KNEE.y - 0.05 and RIGHT_HIP.y > RIGHT_KNEE.y - 0.05:
        activity = "Sitting"

    # Walking
    elif previous_left_ankle_y is not None:
        movement = abs(LEFT_ANKLE.y - previous_left_ankle_y)

        if movement > 0.015:
            activity = "Walking"

    previous_left_ankle_y = LEFT_ANKLE.y

    return activity

# FPS Variables
prev_time = 0
curr_time = 0

# Open webcam
cap = cv2.VideoCapture(0)

print("Press Q to Exit")

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror View
    frame = cv2.flip(frame, 1)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect Pose
    results = pose.process(rgb)

    # Draw Skeleton
    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        activity = detect_activity(landmarks)

        cv2.putText(
            frame,
            f"Activity: {activity}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
)
    # Show Output
    cv2.imshow("Human Activity Recognition AI", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
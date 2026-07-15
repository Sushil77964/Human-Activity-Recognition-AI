import cv2
import mediapipe as mp
import time
import os
from datetime import datetime

# ---------------- Initialize MediaPipe ----------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ---------------- Variables ----------------

previous_left_ankle_y = None

prev_time = 0
curr_time = 0

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


# ---------------- Webcam ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found!")
    exit()

# ---------------- Screenshot Folder ----------------

os.makedirs("screenshots", exist_ok=True)

# ---------------- Video Recording ----------------

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "output.mp4",
    fourcc,
    20,
    (width, height)
)

print("===================================")
print("Press Q or ESC -> Exit")
print("Press S -> Save Screenshot")
print("===================================")

# ---------------- Main Loop ----------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    # FPS
    curr_time = time.time()

    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0

    prev_time = curr_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    activity = "No Person"

    if results.pose_landmarks:

        # Skeleton
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=3),
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
        )

        landmarks = results.pose_landmarks.landmark

        activity = detect_activity(landmarks)

    # Activity
    cv2.putText(
        frame,
        f"Activity : {activity}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    # Time
    current_time = datetime.now().strftime("%I:%M:%S %p")

    cv2.putText(
        frame,
        current_time,
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    # Save Video
    out.write(frame)

    cv2.imshow("Human Activity Recognition AI", frame)

    key = cv2.waitKey(1) & 0xFF

    # Screenshot
    if key == ord("s") or key == ord("S"):

        filename = datetime.now().strftime(
            "screenshots/%Y%m%d_%H%M%S.jpg"
        )

        cv2.imwrite(filename, frame)

        print(f"Screenshot Saved -> {filename}")

    # Exit
    if key == ord("q") or key == ord("Q") or key == 27:
        break


# ---------------- Cleanup ----------------

cap.release()
out.release()
cv2.destroyAllWindows()
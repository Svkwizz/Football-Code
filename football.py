import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video file
cap = cv2.VideoCapture("E:\Research\Clinical video\Topcorner.mp4")

# Initialize variables to store leg trajectory data
left_leg_trajectory = []
right_leg_trajectory = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect poses in the frame
    results = pose.process(frame_rgb)

    # If poses are detected
    if results.pose_landmarks:
        # Extract landmark points
        landmarks = results.pose_landmarks.landmark

        # Get key points for the left leg
        left_ankle = np.array(
            [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y])

        # Get key points for the right leg
        right_ankle = np.array(
            [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y])

        # Calculate trajectory of the right leg
        right_leg_trajectory.append(right_ankle)

    # Display frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert lists to NumPy arrays
left_leg_trajectory = np.array(left_leg_trajectory)
right_leg_trajectory = np.array(right_leg_trajectory)

# Plot trajectories of both legs
plt.figure(figsize=(10, 6))
plt.plot(right_leg_trajectory[:, 0], right_leg_trajectory[:, 1], label='Right Leg Trajectory', color='red')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Trajectories of Both Legs during Penalty Kick')
plt.legend()
plt.grid(True)
plt.show()

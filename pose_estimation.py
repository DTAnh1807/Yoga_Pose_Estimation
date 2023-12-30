import cv2
import mediapipe as mp

def pose_image(image_path):
    # Initialize Mediapipe Pose object
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1)

    # Read the image from the given path
    image = cv2.imread(image_path)

    # Convert the image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation on the image
    results = pose.process(rgb_image)

    # Display the result on the image
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image
    cv2.imshow('Pose Estimation - Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pose_video(video_path):
    # Initialize Mediapipe Pose object
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize VideoCapture object to read the video from the file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation on the frame
        results = pose.process(rgb_frame)

        # Display the result on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Pose Estimation - Video', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

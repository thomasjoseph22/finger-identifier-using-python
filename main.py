import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Initially set finger names to an empty list for each hand
        finger_names = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand index to check label (left or right)
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                # Set variable to keep landmarks positions (x and y)
                handLandmarks = []

                # Fill list with x and y positions of each landmark
                for landmarks in hand_landmarks.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])

                # Identify raised fingers and append their names to the list
                raised_fingers = []
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    raised_fingers.append("Thumb")
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    raised_fingers.append("Thumb")

                if handLandmarks[8][1] < handLandmarks[6][1]:
                    raised_fingers.append("Index Finger")
                if handLandmarks[12][1] < handLandmarks[10][1]:
                    raised_fingers.append("Middle Finger")
                if handLandmarks[16][1] < handLandmarks[14][1]:
                    raised_fingers.append("Ring Finger")
                if handLandmarks[20][1] < handLandmarks[18][1]:
                    raised_fingers.append("Pinky Finger")

                # Combine the names of raised fingers into a string
                finger_names.append(', '.join(raised_fingers))

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Display finger names
        for i, names in enumerate(finger_names):
            cv2.putText(image, names, (50, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display image
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()

#cv2 = OpenCV for image/video handling.
#mediapipe = ML-based framework that can detect face, pose, and hand landmarks.
import cv2
import mediapipe as mp
#Opens your default webcam (device 0).
cap = cv2.VideoCapture(0)
#mp_hands → gives access to MediaPipe’s hand landmark model.
#mp_drawing → utility for drawing landmarks and connections on the image.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#Creates a hand detection model object.
#min_detection_confidence=0.8: requires 80% certainty before saying a hand is present.
#min_tracking_confidence=0.5: requires 50% certainty to keep tracking landmarks across frames.
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
#These are the landmark indices of the fingertips in MediaPipe’s hand model:
'''
4 → Thumb tip
8 → Index finger tip
12 → Middle finger tip
16 → Ring finger tip
20 → Little finger tip
'''
tipIds = [4, 8, 12, 16, 20]

# Define a function to count open fingers for the first detected hand
def countFingers(image, hand_landmarks, handNo=0):
    
    if hand_landmarks:
        # Get all Landmarks of the FIRST Hand VISIBLE
        landmarks = hand_landmarks[handNo].landmark

        # Count Fingers        
        fingers = []
        for lm_index in tipIds:
                # Get Finger Tip and Bottom y Position Value
                finger_tip_y = landmarks[lm_index].y 
                finger_bottom_y = landmarks[lm_index - 2].y #second joint below the tip

                # Check if ANY FINGER is OPEN or CLOSED
                if lm_index !=4:   # Skip Thumb
                    if finger_tip_y < finger_bottom_y: #If tip is above the lower joint
                        fingers.append(1)   # Open
                        print("FINGER with id ",lm_index," is Open")

                    if finger_tip_y > finger_bottom_y: #If tip is below
                        fingers.append(0)   # Closed
                        print("FINGER with id ",lm_index," is Closed")
        #Counts total open fingers
        totalFingers = fingers.count(1)
        # Display Text
        text = f'Fingers: {totalFingers}'
        #Writes the number on the video frame in blue text at position (50,50)
        #1:Font scale (size of the text)
        #2:Thickness of the text stroke (outline width).
        cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

# Define a function to 
def drawHandLanmarks(image, hand_landmarks):

    # Draw connections between landmark points
    if hand_landmarks:

      for landmarks in hand_landmarks:
               
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)


while True:
    success, image = cap.read()

    image = cv2.flip(image, 1)
    
    # Detect the Hands Landmarks 
    results = hands.process(image)

    # Get landmark position from the processed result
    hand_landmarks = results.multi_hand_landmarks

    # Draw Landmarks
    drawHandLanmarks(image, hand_landmarks)

    # Get Hand Fingers Position        
    countFingers(image, hand_landmarks)

    cv2.imshow("Output", image)

    # Quit the window on pressing Spacebar key
    key = cv2.waitKey(1)
    if key == 32:
        break
cap.release()
cv2.destroyAllWindows()

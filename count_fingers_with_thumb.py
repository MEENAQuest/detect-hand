import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

#To load the hands and the drawing_utils module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

#IDs(fingertip) of all the fingers and the thumb
tipIds = [4, 8, 12, 16, 20]

# Define a function to count fingers

def countFingers(image, hand_landmarks, handNo=0): #handNo = 0 means we are only detecting 1 hand.
    
    if hand_landmarks: #checking if hand is visible or not
        # Get all Landmarks of the FIRST Hand VISIBLE
        landmarks = hand_landmarks[handNo].landmark
        # print(landmarks)

        # Count Fingers        
        fingers = []

        for lm_index in tipIds:
                # Get Finger Tip and Bottom y Position Value
                finger_tip_y = landmarks[lm_index].y 
                finger_bottom_y = landmarks[lm_index - 2].y
                
                # Get Thumb Tip and Bottom y Position Value
                thumb_tip_x = landmarks[lm_index].x
                thumb_bottom_x = landmarks[lm_index - 2].x

                # Check if ANY FINGER is OPEN or CLOSED
                if lm_index !=4: #Landmark ID of THUMB is 4
                    if finger_tip_y < finger_bottom_y:
                        fingers.append(1)
                        print("FINGER with id ",lm_index," is Open")

                    if finger_tip_y > finger_bottom_y:
                        fingers.append(0)
                        print("FINGER with id ",lm_index," is Closed")
                else:
                    if thumb_tip_x < thumb_bottom_x:
                        fingers.append(1)
                        print("THUMB is Open")

                    if thumb_tip_x > thumb_bottom_x:
                        fingers.append(0)
                        print("THUMB is Closed")


        # print(fingers)
        totalFingers = fingers.count(1)

        # Display Text
        text = f'Fingers: {totalFingers}'

        cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

# Define a function to 
def drawHandLanmarks(image, hand_landmarks):

    # Darw connections between landmark points
    if hand_landmarks: #checking if hand is visible or not

      for landmarks in hand_landmarks:
               
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)


while True:
    success, image = cap.read()

    #flip image horizontally(1) or vertically(0)
    image = cv2.flip(image, 1)
    
    # Detect the Hands Landmarks 
    results = hands.process(image)

    # Get landmark position from the processed result
    hand_landmarks = results.multi_hand_landmarks

    # Draw Landmarks
    drawHandLanmarks(image, hand_landmarks)

    # Get Hand Fingers Position        
    countFingers(image, hand_landmarks)

    cv2.imshow("Media Controller", image)

    # Quit the window on pressing Sapcebar key 
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()

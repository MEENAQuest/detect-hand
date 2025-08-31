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

    # Get Hand Fingers Position        
    if hand_landmarks:
        # Get all Landmarks of the FIRST Hand VISIBLE
        landmarks = hand_landmarks[0].landmark

    # Draw Landmarks
    drawHandLanmarks(image, hand_landmarks)

    cv2.imshow("Output", image)

    # Quit the window on pressing Spacebar key
    key = cv2.waitKey(1)
    if key == 32:
        break
cap.release()
cv2.destroyAllWindows()
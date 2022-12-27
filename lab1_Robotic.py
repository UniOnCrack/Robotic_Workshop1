#This code demonstrate how to show location of hand landmark
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

#Call hand pipe line module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    finger = []
    finger_axis = np.zeros((21, 2))
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                finger_axis[id] = [cx, cy] 
            if finger_axis[4, 0] > finger_axis[3, 0]:
                finger.append("Thumbs")

            if finger_axis[8, 1] < finger_axis[7, 1]:
                finger.append("Index")

            if finger_axis[12, 1] < finger_axis[11, 1]:
                finger.append("Middle")

            if finger_axis[16, 1] < finger_axis[15, 1]:
                finger.append("Ring")
                
            if finger_axis[20, 1] < finger_axis[19, 1]:
                finger.append("Pinky")

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    Nfing = len(finger)
    cv2.putText(img, str(Nfing), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 252, ), 2)

    cv2.putText(img, str(finger), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 252), 2)

    cv2.imshow("Finger Reconization", img)
    cv2.waitKey(1)
#Closeing all open windows
#cv2.destroyAllWindows()
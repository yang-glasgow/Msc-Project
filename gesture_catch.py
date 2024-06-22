import cv2
import mediapipe as mp
import time
import os
import pandas as pd

# webcam setting
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)

pTime = 0
startTime = time.time()
currentTime = 0

hand_index = 0
data_list = []

if os.path.isfile('GestureDataset.csv'):
    tem_df = pd.read_csv('GestureDataset.csv')
    hand_index = tem_df.iloc[-1, 0] + 1


while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        imgX = img.shape[0]
        imgY = img.shape[1]
        # imgZ = img.shape[2]
        if cv2.waitKey(1) == ord(' '):
            currentTime = time.time()
            if currentTime - startTime >= 0.2:
                startTime = currentTime

                if result.multi_hand_landmarks:
                    for handLms in result.multi_hand_landmarks:
                        for i, lm in enumerate(handLms.landmark):
                            data_list.append([hand_index,i,lm.x,lm.y,lm.z])
                            # print(i, lm.x, lm.y, lm.z)
                        hand_index += 1

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, f"DATA NUMBER : {hand_index}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

# dataframe store the gesture data
columns = ['hand_index', 'landmark_index', 'x', 'y', 'z']
dataset_df = pd.DataFrame(data_list, columns=columns)
if os.path.isfile('GestureDataset.csv'):
    dataset_df.to_csv('GestureDataset.csv', mode='a', index=False, header=False)
else:
    dataset_df.to_csv('GestureDataset.csv', mode='w', index=False, header=True)

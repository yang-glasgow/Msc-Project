import cv2
import mediapipe as mp
import time
import os
import pandas as pd
import torch
from networks import GestureCNN
import torch.nn


def is_shape(l, rows, columns):
    if len(l) != rows:
        return False
    for row in l:
        if len(row) != columns:
            return False
    return True


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

gesture_model = GestureCNN()
gesture_model.load_state_dict(torch.load('gestureModelState.pth', map_location=torch.device('cpu')))
gesture_model.eval()

hand_list = []
pred_label = 0

while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        imgX = img.shape[0]
        imgY = img.shape[1]
        # imgZ = img.shape[2]

        hand_list = []

        currentTime = time.time()
        if currentTime - startTime >= 0.5:
            startTime = currentTime
            if result.multi_hand_landmarks:
                handLms = result.multi_hand_landmarks[0]
                for i, lm in enumerate(handLms.landmark):
                    hand_list.append([lm.x, lm.y, lm.z])

                if is_shape(hand_list, 21, 3):
                    hand_input = torch.tensor(hand_list)
                    hand_input = torch.unsqueeze(hand_input, 0)
                    output = gesture_model(hand_input.float())
                    pred_label = output.argmax()

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, f"Predicted Labelq : {pred_label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

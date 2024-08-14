import cv2
import mediapipe as mp
import time
import os
import pandas as pd

from tkinter import *
from tkinter.ttk import *


class GetNumberLabel:
    def __init__(self):
        self.window = Tk()
        self.window.title("Gesture Capture")
        w_screen = self.window.winfo_screenwidth()
        h_screem = self.window.winfo_screenheight()
        w_window = 400
        h_window = 150
        minw = (w_screen - w_window) / 2
        minh = (h_screem - h_window) / 2
        self.window.geometry("%dx%d+%d+%d" % (w_window, h_window, minw, minh))
        self.window.resizable(False, False)

        self.gestures_numbers = 0
        self.gesture_label = 99

        label_gestures_numbers = Label(self.window, text="Number of gestures to be captured:")
        label_gestures_numbers.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.entry_gestures_numbers = Entry(self.window)
        self.entry_gestures_numbers.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        label_gesture_label = Label(self.window, text="Label of these gestures:")
        label_gesture_label.grid(row=1, column=0, padx=10, pady=10)

        self.entry_gesture_label = Entry(self.window)
        self.entry_gesture_label.grid(row=1, column=1, padx=10, pady=10)

        submit_button = Button(self.window, text="Submit", command=self.on_submit)
        submit_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def on_submit(self):
        self.gestures_numbers = self.entry_gestures_numbers.get()
        self.gesture_label = self.entry_gesture_label.get()

        if self.gestures_numbers is None:
            pass
        else:
            self.gestures_numbers = int(self.gestures_numbers)

        if self.gesture_label is None:
            pass
        else:
            self.gesture_label = int(self.gesture_label)

        self.window.destroy()

    def get_inputs(self):
        self.window.mainloop()
        return self.gestures_numbers, self.gesture_label


class ConfirmContinue:
    def __init__(self):
        self.window = Tk()
        self.window.title("If Continue Gesture Collection?")

        w_screen = self.window.winfo_screenwidth()
        h_screen = self.window.winfo_screenheight()
        w_window = 400
        h_window = 150
        minw = (w_screen - w_window) / 2
        minh = (h_screen - h_window) / 2
        self.window.geometry("%dx%d+%d+%d" % (w_window, h_window, minw, minh))
        self.window.resizable(False, False)

        self.catch_continue = True

        label_confirm = Label(self.window, text="Continue collecting gestures?")
        label_confirm.grid(row=0, column=0, columnspan=2, padx=10, pady=20)

        yes_button = Button(self.window, text="Yes", command=self.choose_yes)
        yes_button.grid(row=1, column=0, padx=20, pady=20, sticky="ew")

        no_button = Button(self.window, text="No", command=self.choose_no)
        no_button.grid(row=1, column=1, padx=20, pady=20, sticky="ew")

        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)

    def choose_yes(self):
        self.catch_continue = True
        self.window.destroy()

    def choose_no(self):
        self.catch_continue = False
        self.window.destroy()

    def get_confirmation(self):
        self.window.mainloop()
        return self.catch_continue


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
catch_continue = True

file_name = 'Dataset.csv'

while catch_continue:
    gesture_count = 0
    hand_index = 0
    data_list = []

    if os.path.isfile(file_name):
        tem_df = pd.read_csv(file_name)
        hand_index = tem_df.iloc[-1, 0] + 1

    gesture_number = 0
    gesture_label = 99

    get_number_label = GetNumberLabel()
    gesture_number, gesture_label = get_number_label.get_inputs()

    while True:
        if gesture_count is gesture_number:
            break

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
                                data_list.append([hand_index, i, lm.x, lm.y, lm.z, gesture_label])
                                # print(i, lm.x, lm.y, lm.z)

                        hand_index += 1
                        gesture_count += 1

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime
            # cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f"Gesture Captured : {gesture_count}", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        3)
            cv2.putText(img, f"Gesture Index : {hand_index}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        3)
            cv2.imshow('img', img)

    dataset_df = pd.DataFrame(data_list,columns=['hand_index','landmark_index','x','y','z','label'])
    if os.path.isfile(file_name):
        dataset_df.to_csv(file_name, mode='a', index=False, header=False)
    else:
        dataset_df.to_csv(file_name, mode='w', index=False, header=True)

    confirmContinue = ConfirmContinue()
    catch_continue = confirmContinue.get_confirmation()

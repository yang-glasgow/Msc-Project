import cv2
import mediapipe as mp
import time
import os
import pandas as pd
import torch
from networks import GestureCNN
import torch.nn
import torch
import sys
import sounddevice as sd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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

#initialise gesture CNN model
model_parameters = {'c1': 10, 'c2': 62, 'l1': 487, 'ks': 5}
gesture_model = GestureCNN(**model_parameters)
gesture_model.load_state_dict(torch.load('gestureModelState.pth'))
gesture_model.to(device)
gesture_model.eval()

hand_list = []
pred_label = 0

# initialise the WaveGlow
waveglow_path = './DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/waveglow/'
sys.path.append(waveglow_path)
tacotron2_path = './DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/'
sys.path.append(tacotron2_path)

from denoiser import Denoiser
from model import WaveGlow
import data_function
import loss_function
from loss_function import WaveGlowLoss
import importlib.util
from entrypoints import nvidia_waveglow

entrypoints_path = './DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/waveglow/entrypoints.py'

spec = importlib.util.spec_from_file_location("entrypoints", entrypoints_path)
entrypoints = importlib.util.module_from_spec(spec)
sys.modules["entrypoints"] = entrypoints
spec.loader.exec_module(entrypoints)

waveglow = nvidia_waveglow(pretrained=True, model_math='fp32')
waveglow = waveglow.cuda()

def play_mel_audio(mel_spectrogram):
    with torch.no_grad():
        audio = waveglow.infer(mel_spectrogram.unsqueeze(0).to(device), sigma=1)

    denoiser = Denoiser(waveglow)
    audio_denoised = denoiser(audio, strength=0.01)[:, 0].to('cpu')
    sd.play(audio_denoised[0], samplerate=18000)
    sd.wait()

    return 1

#intialise Tacotron2
from tacotron2.model import Tacotron2
from tacotron2.loss_function import Tacotron2Loss
hparams = {
    'n_mel_channels': 80,
    'n_symbols': 148,
    'symbols_embedding_dim': 512,
    'encoder_kernel_size': 5,
    'encoder_n_convolutions': 3,
    'encoder_embedding_dim': 512,
    'attention_rnn_dim': 1024,
    'attention_dim': 128,
    'attention_location_n_filters': 32,
    'attention_location_kernel_size': 31,
    'n_frames_per_step': 1,
    'decoder_rnn_dim': 1024,
    'prenet_dim': 256,
    'max_decoder_steps': 1000,
    'gate_threshold': 0.5,
    'p_attention_dropout': 0.1,
    'p_decoder_dropout': 0.1,
    'postnet_embedding_dim': 512,
    'postnet_kernel_size': 5,
    'postnet_n_convolutions': 5,
    'decoder_no_early_stopping': False,
    'mask_padding': False
}
model = Tacotron2(**hparams).to(device)
checkpoint = torch.load(os.path.join('./pretrained_check_points/','checkpoint_epoch_640.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def phoneme_sequence_to_speech(inputs):
    length = [len(inputs)]
    inputs = torch.tensor(inputs, dtype=torch.int).unsqueeze(0).to(device)
    input_lengths = torch.tensor(length, dtype=torch.int).to(device)

    with torch.no_grad():
        mel_outputs, mel_lengths, alignments = model.infer(inputs, input_lengths)
        mel_outputs, mel_lengths, alignments = model.infer(inputs, input_lengths)

    play_mel_audio(mel_outputs[0])

#phoneme_to_index and reverse mapping
pho39_to_index = {
    'aa': 1, 'ae': 2, 'ah': 3, 'aw': 4, 'ay': 5, 'eh': 6, 'er': 7, 'ey': 8,
    'dh': 9, 'dx': 10, 'b': 11, 'd': 12, 'ch': 13, 'f': 14, 'g': 15, 'z': 16,
    'hh': 17, 'ih': 18, 'iy': 19, 'jh': 20, 'k': 21, 'l': 22, 'm': 23, 'n': 24,
    'ng': 25, 'ow': 26, 'oy': 27, 'p': 28, 'r': 29, 's': 30, 'sh': 31, 't': 32,
    'th': 33, 'uh': 34, 'uw': 35, 'v': 36, 'w': 37, 'y': 38, 'h#': 39
}
index_to_pho39 = {value: key for key, value in pho39_to_index.items()}

pTime = 0
startTime = time.time()
currentTime = 0
pho_sequence = ['h#']
input_sequence = [39]
last_label = 0
prev_label = 0
start_convert = False

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

        if start_convert:
            start_convert = False
            input_sequence = [x - 1 for x in input_sequence]
            phoneme_sequence_to_speech(input_sequence)
            input_sequence = [39]
            pho_sequence = ['h#']

        currentTime = time.time()
        if currentTime - startTime >= 0.5:
            startTime = currentTime
            if result.multi_hand_landmarks:
                handLms = result.multi_hand_landmarks[0]
                for i, lm in enumerate(handLms.landmark):
                    hand_list.append([lm.x, lm.y, lm.z])

                if is_shape(hand_list, 21, 3):
                    hand_input = torch.tensor(hand_list)
                    hand_input = torch.unsqueeze(hand_input, 0).to(device)
                    output = gesture_model(hand_input.float())
                    pred_label = output.argmax()

                if last_label == prev_label:
                    prev_label = last_label
                    last_label = pred_label
                    continue
                if last_label != prev_label:
                    if last_label != pred_label:
                        prev_label = last_label
                        last_label = pred_label
                        continue

                    prev_label = last_label
                    last_label = pred_label
                    if pred_label != 0 and pred_label != 40:
                        input_sequence.append(int(pred_label))
                        pho_sequence.append(index_to_pho39[int(pred_label)])
                    elif pred_label == 40:
                        input_sequence.append(int(pred_label))
                        cv2.putText(img, f"Speaking", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 3)
                        start_convert = True


        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        # cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, f"Predicted Label : {pred_label}", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, f"Phoneme: {pho_sequence}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

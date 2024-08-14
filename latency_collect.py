import time
import os
import pandas as pd
from networks import GestureCNN
import torch.nn
import torch
import sys
from torch.utils.data import Dataset, DataLoader, random_split
import random
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class GestureDataset(Dataset):

    def __init__(self, file, transform=None):

        self.train_frame = pd.read_csv(file)

        data = self.train_frame[['x', 'y', 'z']].values

        gestures = []
        labels = []

        num_points_per_gesture = 21

        for i in range(0, len(data), num_points_per_gesture):
            gesture = data[i:i + num_points_per_gesture]
            gestures.append(gesture)
            labels.append(self.train_frame.iloc[i, 5])

        self.gestures_data = gestures
        self.labels = labels

    def __len__(self):
        return len(self.gestures_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_values = self.gestures_data[idx]
        label = self.labels[idx]

        return torch.tensor(input_values).to(device), torch.tensor(label).to(device)

gestureDataset = GestureDataset('./GestureDataset.csv')
train_loader = DataLoader(gestureDataset, batch_size=1, shuffle=True)

#initialise gesture CNN model
model_parameters = {'c1': 10, 'c2': 62, 'l1': 487, 'ks': 5}
gesture_model = GestureCNN(**model_parameters)
gesture_model.load_state_dict(torch.load('gestureModelState.pth'))
gesture_model.to(device)
gesture_model.eval()

gesture_model_latency = []

for inputs, labels in train_loader:
    start_time = time.perf_counter()
    outputs = gesture_model(inputs.to(device).float())
    end_time = time.perf_counter()
    gesture_model_latency.append(end_time - start_time)

with open('gesture_model_latency.pkl', 'wb') as f:
    pickle.dump(gesture_model_latency, f)

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

    return 1

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

tacotron_latency = []

for phoneme_len in range(3, 29, 5):
    for i in range(300):
        random_list = [random.randint(0, 37) for _ in range(phoneme_len)]
        inputs = [38] + random_list + [38, 39]

        start_time = time.perf_counter()
        length = [len(inputs)]
        inputs = torch.tensor(inputs, dtype=torch.int).unsqueeze(0).to(device)
        input_lengths = torch.tensor(length, dtype=torch.int).to(device)

        with torch.no_grad():
            mel_outputs, mel_lengths, alignments = model.infer(inputs, input_lengths)
        play_mel_audio(mel_outputs[0])

        end_time = time.perf_counter()
        tacotron_latency.append(end_time - start_time)
    print(f'phoneme lengths : {phoneme_len + 2} has finished')

with open('tacotron2_WaveGlow_model_latency.pkl', 'wb') as f:
    pickle.dump(tacotron_latency, f)
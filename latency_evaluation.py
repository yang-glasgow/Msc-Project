import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open('gesture_model_latency.pkl', 'rb') as f:
    gesture_model_latency = pickle.load(f)

mean_gesture_latency = np.mean(gesture_model_latency)
median_gesture_latency = np.median(gesture_model_latency)
std_gesture_latency = np.std(gesture_model_latency)
min_gesture_latency = np.min(gesture_model_latency)
max_gesture_latency = np.max(gesture_model_latency)

print("Mean:", mean_gesture_latency)
print("Median:", median_gesture_latency)
print("Standard Deviation:", std_gesture_latency)
print("Min:", min_gesture_latency)
print("Max:", max_gesture_latency)

gesture_array = np.array(gesture_model_latency)
filtered_gesture = gesture_array[(gesture_array >= 0.0003) & (gesture_array <= 0.0007)]
print(f'excluded gestures: {len(gesture_array) - len(filtered_gesture)}')

plt.figure(figsize=(10, 5))
sns.histplot(filtered_gesture, bins=40, kde=True, color='blue')
plt.title('CNN Latency Histogram')
plt.xlabel('Latency (seconds)')
plt.ylabel('Frequency')
plt.savefig('CNN_latency')
plt.show()

with open('tacotron2_WaveGlow_model_latency.pkl', 'rb') as f:
    tacotron_latency = pickle.load(f)

speak_latency = [[] for _ in range(6)]
for i in range(6):
    for j in range(0,300):
        latency = tacotron_latency[i*300 + j]
        speak_latency[i].append(latency)

sequence_lengths = [5, 10, 15, 20, 25, 30]
for i in range(6):
    mean_gesture_latency = np.mean(speak_latency[i])
    median_gesture_latency = np.median(speak_latency[i])
    std_gesture_latency = np.std(speak_latency[i])
    min_gesture_latency = np.min(speak_latency[i])
    max_gesture_latency = np.max(speak_latency[i])

    print(f'For sequence length {sequence_lengths[i]}')
    print("Mean:", mean_gesture_latency)
    print("Median:", median_gesture_latency)
    print("Standard Deviation:", std_gesture_latency)
    print("Min:", min_gesture_latency)
    print("Max:", max_gesture_latency)

plt.figure(figsize=(10, 6))
sns.boxplot(data=speak_latency)
plt.xticks(ticks=range(len(sequence_lengths)), labels=sequence_lengths)
plt.title('Latency of Speech Synthesis Model')
plt.xlabel('Phoneme Sequence Length')
plt.ylabel('Latency (seconds)')
plt.grid(True)
plt.savefig('speech_model_latency')
plt.show()

pho5_latency = [33.97, 23.45, 19.14, 15.31, 17.32, 12.59, 9.46, 10.33, 9.17, 7.53]

pho9_latency = [37.53, 25.75, 21.23, 19.32, 16.60, 14.31, 18.22, 13.03, 13.58, 13.95]

pho11_latency = [61.43, 38.30, 52.68, 38.19, 32.77, 38.87, 39.32, 40.97, 26.76, 31.25]

epochs = range(1, 11)

plt.figure(figsize=(10, 6))

plt.plot(epochs, pho5_latency, label='5 Phonemes', color='red', marker='.')
plt.plot(epochs, pho9_latency, label='9 Phonemes', color='blue', marker='.')
plt.plot(epochs, pho11_latency, label='11 Phonemes', color='green', marker='.')

plt.title('User Input Latency')
plt.xlabel('Input Number')
plt.ylabel('Latency (seconds)')

plt.legend()
plt.savefig('user_input_latency.png')
plt.show()

data = {
    "Ease of Learning": [2, 1, 2, 1, 4, 3, 3],
    "User-Friendliness": [4, 3, 2, 5, 5, 4, 3],
    "Response Speed": [3, 1, 3, 1, 2, 4, 2],
    "Accuracy": [4, 3, 4, 2, 3, 4, 5],
    "Reliability": [4, 4, 3, 1, 2, 4, 3],
    "Clarity": [3, 4, 3, 1, 2, 5, 4],
    "Overall Rating": [3, 3, 3, 1, 3, 4, 4]
}

df = pd.DataFrame(data)

descriptive_stats = df.describe()
print(descriptive_stats.loc['mean'])

plt.figure(figsize=(8, 4))

sns.boxplot(data=df)
plt.title('User Feedback')
plt.ylabel('Ratings')
plt.xticks(rotation = 10)
plt.grid(True)
plt.tight_layout()

plt.savefig('user_feedback.png')
plt.show()
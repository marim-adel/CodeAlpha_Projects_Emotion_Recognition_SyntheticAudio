# Generate tiny synthetic audio dataset (sine waves) for 3 'emotions' and train a small model.
import os, numpy as np, joblib
import soundfile as sf
from sklearn.model_selection import train_test_split
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
emotions = ['happy','sad','angry']
os.makedirs('data/audio', exist_ok=True)
sr = 22050
durations = [1.5, 1.5, 1.5]
# create simple sine waves with different freq patterns as proxies for emotions
freqs = [880, 220, 440]
files = []
for i, e in enumerate(emotions):
    for j in range(30):  # 30 samples each -> small dataset
        t = np.linspace(0, durations[i], int(sr*durations[i]), False)
        # vary frequency slightly
        f = freqs[i] + np.random.randn()*10
        wave = 0.5*np.sin(2*np.pi*f*t)
        # add small noise and modulation
        wave += 0.02*np.random.randn(len(wave))
        path = f'data/audio/{e}_{j}.wav'
        sf.write(path, wave, sr)
        files.append((path, i))
# extract MFCCs
def extract_mfcc(path, n_mfcc=20, max_len=66):
    y, sr = librosa.load(path, sr=sr)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mf.shape[1] < max_len:
        pad = max_len - mf.shape[1]
        mf = np.pad(mf, ((0,0),(0,pad)))
    else:
        mf = mf[:,:max_len]
    return mf
X = []
y = []
for p, lab in files:
    mf = extract_mfcc(p)
    X.append(mf)
    y.append(lab)
X = np.array(X)  # shape (N, n_mfcc, time)
X = X[..., np.newaxis]  # add channel
y = np.array(y)
from tensorflow.keras.utils import to_categorical
ycat = to_categorical(y, num_classes=3)
# simple CNN
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=X.shape[1:]),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, ycat, epochs=12, batch_size=16, validation_split=0.1)
os.makedirs('models', exist_ok=True)
model.save('models/emotion_cnn_tf')
print('Saved model to models/emotion_cnn_tf')
# save label map
joblib.dump({'labels': emotions}, 'models/emotion_meta.joblib')

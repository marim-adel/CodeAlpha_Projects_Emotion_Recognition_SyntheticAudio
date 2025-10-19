import tensorflow as tf, numpy as np, joblib, librosa
def load_model(path='models/emotion_cnn_tf'):
    m = tf.keras.models.load_model(path)
    meta = joblib.load('models/emotion_meta.joblib')
    return m, meta['labels']
def predict_audio(model, labels, wav_path):
    import numpy as np, librosa
    sr = 22050
    y, _ = librosa.load(wav_path, sr=sr)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    max_len = 66
    if mf.shape[1] < max_len:
        mf = np.pad(mf, ((0,0),(0,max_len-mf.shape[1])))
    else:
        mf = mf[:,:max_len]
    arr = mf[np.newaxis,...,np.newaxis]
    preds = model.predict(arr)
    idx = int(preds.argmax(axis=1)[0])
    return labels[idx], float(preds.max())

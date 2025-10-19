import streamlit as st, joblib
st.title('Emotion Recognition - Synthetic Audio Demo')
st.write('This demo trains on tiny synthetic sine-wave audio to illustrate the pipeline.')
if st.button('Train synthetic audio model (small)'):
    import subprocess, sys
    subprocess.run([sys.executable, 'src/train.py'])
    st.success('Training done.')
if st.button('Load model'):
    import tensorflow as tf
    st.session_state['emo_model'] = tf.keras.models.load_model('models/emotion_cnn_tf')
    st.session_state['emo_labels'] = joblib.load('models/emotion_meta.joblib')['labels']
    st.success('Model loaded.')
uploaded = st.file_uploader('Upload WAV file to predict', type=['wav'])
if uploaded is not None and 'emo_model' in st.session_state:
    with open('temp_upload.wav','wb') as f:
        f.write(uploaded.getbuffer())
    model = st.session_state['emo_model']
    labels = st.session_state['emo_labels']
    from src.predict import predict_audio
    lab, conf = predict_audio(model, labels, 'temp_upload.wav')
    st.write('Predicted emotion:', lab, 'conf:', conf)
elif uploaded is not None:
    st.info('Train and load model first (buttons above).')

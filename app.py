import streamlit as st
import numpy as np
import pickle
import re
import string
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Load assets (once) ──────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = load_model('./emotion_bilstm.h5')
    with open('./tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('./label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, tokenizer, encoder

model, tokenizer, encoder = load_assets()

MAX_LEN = 50  # must match what you used during training

# ── Preprocessing (same pipeline as training) ────────────────────────
stop_words = set(stopwords.words('english'))
stop_words -= {'not', 'no', 'nor', 'never', 'very', 'too'}
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()])
    text = ' '.join([w for w in text.split() if w not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

def predict_emotion(text):
    cleaned = preprocess(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model.predict(padded, verbose=0)[0]
    pred_label = encoder.classes_[np.argmax(probs)]
    return pred_label, probs

# ── UI ───────────────────────────────────────────────────────────────
EMOJI = {
    'joy':      '😄',
    'sadness':  '😢',
    'anger':    '😡',
    'fear':     '😨',
    'love':     '❤️',
    'surprise': '😲'
}

st.title("🧠 Emotion Detector")
st.write("Powered by a Bidirectional LSTM trained on the Emotions NLP dataset.")

text_input = st.text_area("Enter a sentence:", placeholder="e.g. I feel so lost and empty today...")

if st.button("Detect Emotion") and text_input.strip():
    with st.spinner("Analyzing..."):
        label, probs = predict_emotion(text_input)

    st.markdown(f"### {EMOJI.get(label, '')} Predicted Emotion: **{label.upper()}**")

    st.write("#### Confidence scores:")
    for i, cls in enumerate(encoder.classes_):
        st.progress(float(probs[i]), text=f"{EMOJI.get(cls, '')} {cls}: {probs[i]*100:.1f}%")
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model
model = load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("Fake News Detector")

# Input
user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=100)

    pred = model.predict(padded)[0][0]

    if pred > 0.5:
        st.success("🟢 Real News")
    else:
        st.error("🔴 Fake News")
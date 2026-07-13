import streamlit as st
import pandas as pd
import pickle

st.title("📰 Fake News Detector")

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# User input
news = st.text_area("Enter News Text:")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some text")
    else:
        # Transform input
        data = vectorizer.transform([news])
        prediction = model.predict(data)[0]

        if prediction == 1:
            st.success("✅ This is TRUE News")
        else:
            st.error("❌ This is FAKE News")

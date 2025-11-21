import streamlit as st
import pickle
import os
import subprocess
import sys

MODEL_FILE = "model.pkl"
VECT_FILE = "vectorizer.pkl"
TRAIN_SCRIPT = "train_model.py"

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection")
st.write("A simple ML model to classify news as real or fake.")

# MAIN INPUT
news_text = st.text_area("Enter News Text Here:", height=200)

# TRAINING SECTION
if st.button("Train Model"):
    with st.spinner("Training model... please wait"):
        result = subprocess.run([sys.executable, TRAIN_SCRIPT], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Training completed! model.pkl and vectorizer.pkl created.")
        else:
            st.error("Training failed.")
            st.code(result.stdout + "\n\n" + result.stderr)

# PREDICTION SECTION
if st.button("Predict"):
    if not news_text:
        st.warning("Please enter text before predicting.")
    else:
        if not os.path.exists(MODEL_FILE) or not os.path.exists(VECT_FILE):
            st.error("Model files not found! Please train first.")
        else:
            model = pickle.load(open(MODEL_FILE, "rb"))
            vectorizer = pickle.load(open(VECT_FILE, "rb"))

            X = vectorizer.transform([news_text])
            prediction = model.predict(X)[0]

            if str(prediction) in ["1", "real", "Real"]:
                st.success("✅ REAL NEWS")
            else:
                st.error("❌ FAKE NEWS")

st.markdown("---")
st.write("Developed using Streamlit + Machine Learning")


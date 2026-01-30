# ---------------- SYSTEM SETUP ----------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuroGraph – Parkinson’s Dashboard",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model(
        "models/cnn_model_clean.h5", compile=False
    )

cnn_model = load_cnn()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧠 NeuroGraph")
    st.markdown("Medical AI Dashboard")
    st.divider()
    st.button("📊 Dashboard")
    st.button("👤 Patients")
    st.button("📈 Analytics")
    st.button("📄 Reports")

# ---------------- IMAGE PROCESSING ----------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary

def cnn_predict(binary_img):
    img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return float(cnn_model.predict(img, verbose=0)[0][0])

# ---------------- MAIN DASHBOARD ----------------
st.markdown("## 📊 Parkinson’s Disease Prediction Dashboard")

col_input, col_result = st.columns([1, 1])

# -------- Patient Input Card --------
with col_input:
    st.markdown("### 🧍 Patient Input")
    uploaded_file = st.file_uploader(
        "Upload Spiral Handwriting Image",
        type=["png", "jpg", "jpeg"]
    )
    predict_btn = st.button("🔍 Predict")

# -------- Prediction Result Card --------
with col_result:
    st.markdown("### 🧠 Prediction Result")

    if uploaded_file and predict_btn:
        img = cv2.imdecode(
            np.frombuffer(uploaded_file.read(), np.uint8), 1
        )
        processed = preprocess_image(img)
        prob = cnn_predict(processed)

        if prob < 0.33:
            severity = "Healthy / Mild"
        elif prob < 0.66:
            severity = "Moderate Parkinson’s"
        else:
            severity = "Severe Parkinson’s"

        st.markdown(f"## **{severity}**")

        # Gauge-style confidence
        fig, ax = plt.subplots()
        ax.barh([0], [prob * 100], color="#2D8CFF")
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Confidence Level (%)")
        ax.set_title(f"{prob*100:.1f}% Confidence")
        st.pyplot(fig)

# ---------------- LOWER SECTION ----------------
col_feat, col_prog = st.columns(2)

# -------- Feature Importance --------
with col_feat:
    st.markdown("### 📌 Feature Importance (Estimated)")
    features = {
        "Path Length": 0.30,
        "Curvature Mean": 0.22,
        "Direction Changes": 0.18,
        "Smoothness": 0.15,
        "Stroke Density": 0.15
    }
    fig, ax = plt.subplots()
    ax.barh(list(features.keys()), list(features.values()), color="#6CB4EE")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

# -------- Disease Progression --------
with col_prog:
    st.markdown("### 📈 Progression Over Time")
    timeline = pd.DataFrame({
        "Month": [0, 6, 12, 18, 24],
        "Severity Score": [1, 2, 1.5, 2.5, 3]
    })
    st.line_chart(timeline.set_index("Month"))

# ---------------- RECENT REPORTS ----------------
st.markdown("### 🗂️ Recent Reports")

reports = pd.DataFrame({
    "Report ID": ["PD-2026-001", "PD-2026-002"],
    "Patient Name": ["John Smith", "Jane Doe"],
    "Prediction": ["Moderate", "Mild"],
    "Date": ["2026-01-29", "2026-01-28"]
})

st.dataframe(reports, use_container_width=True)

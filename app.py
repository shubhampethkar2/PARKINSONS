# ---------------- SYSTEM SETUP ----------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TF GPU warnings

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="NeuroGraph – Parkinson’s Analysis",
    layout="wide"
)

# ---------------- LOAD MODELS (NO STREAMLIT CACHE) ----------------
def load_models():
    cnn = load_model("models/cnn_model_clean.h5", compile=False)  # FIXED
    svm = joblib.load("models/svm_model.pkl")
    rf = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    encoder = joblib.load("models/severity_encoder.pkl")
    return cnn, svm, rf, scaler, encoder

cnn_model, svm_model, rf_model, scaler, severity_encoder = load_models()

# ---------------- NORMAL RANGES ----------------
NORMAL_RANGES = {
    "path_length": (900, 1100),
    "curvature_mean": (0.05, 0.15),
    "curvature_std": (0.1, 0.4),
    "direction_changes": (0, 20),
    "stroke_density": (0.015, 0.020),
    "smoothness": (0.02, 0.06)
}

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return clean

# ---------------- FEATURE EXTRACTION (SAFE) ----------------
def extract_features(binary_img):
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return None  # safety check

    contour = max(contours, key=cv2.contourArea)
    pts = contour.squeeze()

    if len(pts.shape) != 2:
        return None

    diffs = np.diff(pts, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_diff = np.abs(np.diff(angles))

    return {
        "path_length": float(distances.sum()),
        "curvature_mean": float(np.mean(angle_diff)),
        "curvature_std": float(np.std(angle_diff)),
        "direction_changes": int(np.sum(angle_diff > np.pi / 4)),
        "stroke_density": float(len(pts) / binary_img.size),
        "smoothness": float(np.std(distances))
    }

# ---------------- CNN PREDICTION ----------------
def cnn_predict(binary_img):
    img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prob = float(cnn_model.predict(img, verbose=0)[0][0])
    return prob

# ---------------- UI ----------------
st.title("🧠 NeuroGraph")
st.subheader("Explainable AI for Parkinson’s Disease Detection using Handwriting")

uploaded_file = st.file_uploader(
    "Upload Spiral Handwriting Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    if img is None:
        st.error("Invalid image file.")
        st.stop()

    processed = preprocess_image(img)
    features = extract_features(processed)

    if features is None:
        st.error("Unable to extract handwriting features. Please upload a clear spiral image.")
        st.stop()

    feature_df = pd.DataFrame([features])
    X_scaled = scaler.transform(feature_df)

    # ---------------- MODEL PREDICTIONS ----------------
    cnn_prob = cnn_predict(processed)
    svm_pred = svm_model.predict(X_scaled)[0]
    rf_pred = rf_model.predict(X_scaled)[0]

    severity = severity_encoder.inverse_transform([svm_pred])[0]
    final_pd = "Yes" if cnn_prob > 0.5 else "No"

    # ---------------- RESULTS ----------------
    st.markdown("## 📊 Diagnostic Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Parkinson’s Detected", final_pd)
    col2.metric("Severity Level", severity)
    col3.metric("CNN Confidence", f"{cnn_prob:.2f}")

    # ---------------- FEATURE REPORT ----------------
    st.markdown("## 🔍 Feature-wise Clinical Analysis")

    report = []
    for f, val in features.items():
        low, high = NORMAL_RANGES[f]
        status = "Normal" if low <= val <= high else "Abnormal"
        report.append([f, f"{low} – {high}", f"{val:.4f}", status])

    report_df = pd.DataFrame(
        report,
        columns=["Feature", "Normal Range", "Patient Value", "Status"]
    )

    st.dataframe(report_df, use_container_width=True)

    # ---------------- MODEL DECISIONS ----------------
    st.markdown("## 🤖 Model-wise Decisions")

    st.write(f"**CNN:** PD Probability = {cnn_prob:.2f}")
    st.write(f"**SVM:** Severity = {severity}")
    st.write(
        f"**Random Forest:** Severity = {severity_encoder.inverse_transform([rf_pred])[0]}"
    )

    # ---------------- DOWNLOAD REPORT ----------------
    st.download_button(
        "📄 Download Full Report (CSV)",
        report_df.to_csv(index=False),
        file_name="neurograph_report.csv"
    )

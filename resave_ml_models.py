import joblib

# Load old models
svm = joblib.load("models/svm_model.pkl")
rf = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
encoder = joblib.load("models/severity_encoder.pkl")

# Re-save models in clean NumPy environment
joblib.dump(svm, "models/svm_model_clean.pkl")
joblib.dump(rf, "models/rf_model_clean.pkl")
joblib.dump(scaler, "models/feature_scaler_clean.pkl")
joblib.dump(encoder, "models/severity_encoder_clean.pkl")

print("✅ SVM, RF, scaler, and encoder re-saved successfully")

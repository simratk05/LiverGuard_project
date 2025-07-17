import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and scaler
stacked_model = joblib.load("stacked_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Liver Health Predictor", layout="centered")
st.title("🧬 Liver Health Prediction (LiverGuard)")
st.markdown("Enter the real-time sensor data below:")

with st.form("sensor_form"):
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    r = st.number_input("Red Value (R)")
    g = st.number_input("Green Value (G)")
    b = st.number_input("Blue Value (B)")
    c = st.number_input("Intensity Value (C)")
    body_temp = st.number_input("Body Temperature (°C)")
    liver_temp = st.number_input("Liver Temperature (°C)")
    gsr = st.number_input("GSR")
    bmi = st.number_input("BMI")
    submit = st.form_submit_button("Predict")

def compute_yellowness_index(r, g, b, c):
    rgb = np.array([[r, g, b]], dtype=float)
    C_array = np.array([[c]])
    rgb_norm = rgb / np.clip(C_array, 1e-6, None)

    gray_world_avg = np.mean(rgb_norm, axis=0)
    rgb_balanced = np.clip(rgb_norm / (gray_world_avg + 1e-6), 0, 1)

    gamma = 2.2
    rgb_linear = np.power(rgb_balanced, gamma)

    M_sRGB_D65 = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = rgb_linear @ M_sRGB_D65.T
    X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    Cx, Cz = 1.2769, 1.0592
    YI_raw = 100 * (Cx * X - Cz * Z) / np.clip(Y, 1e-6, None)
    YI_norm = (YI_raw - YI_raw.min()) / (YI_raw.max() - YI_raw.min() + 1e-6)
    return YI_norm[0]

if submit:
    try:
        gender_val = 1.0 if gender == "Male" else 0.0
        yi = compute_yellowness_index(r, g, b, c)

        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender_val,
            "BodyTemp": body_temp,
            "LiverTemp": liver_temp,
            "GSR": gsr,
            "BMI": bmi,
            "Yellowness Index": yi
        }])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        pred = stacked_model.predict(input_scaled)[0]
        result = "🟢 Healthy" if pred == 0 else "🔴 Unhealthy"

        st.subheader("Prediction Result")
        st.success(result)

    except Exception as e:
        st.error(f"Prediction failed: {e}")



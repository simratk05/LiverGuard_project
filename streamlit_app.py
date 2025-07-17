# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load models and scaler
# stacked_model = joblib.load("stacked_model.pkl")
# scaler = joblib.load("scaler.pkl")

# # Streamlit UI
# st.set_page_config(page_title="Liver Health Predictor", layout="centered")
# st.title("🧬 Liver Health Prediction (LiverGuard)")
# st.markdown("Enter the real-time sensor data below:")

# with st.form("sensor_form"):
#     age = st.number_input("Age", min_value=1, max_value=100, value=30)
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     r = st.number_input("Red Value (R)")
#     g = st.number_input("Green Value (G)")
#     b = st.number_input("Blue Value (B)")
#     c = st.number_input("Intensity Value (C)")
#     body_temp = st.number_input("Body Temperature (°C)")
#     liver_temp = st.number_input("Liver Temperature (°C)")
#     gsr = st.number_input("GSR")
#     bmi = st.number_input("BMI")
#     submit = st.form_submit_button("Predict")

# def compute_yellowness_index(r, g, b, c):
#     rgb = np.array([[r, g, b]], dtype=float)
#     C_array = np.array([[c]])
#     rgb_norm = rgb / np.clip(C_array, 1e-6, None)

#     gray_world_avg = np.mean(rgb_norm, axis=0)
#     rgb_balanced = np.clip(rgb_norm / (gray_world_avg + 1e-6), 0, 1)

#     gamma = 2.2
#     rgb_linear = np.power(rgb_balanced, gamma)

#     M_sRGB_D65 = np.array([
#         [0.4124564, 0.3575761, 0.1804375],
#         [0.2126729, 0.7151522, 0.0721750],
#         [0.0193339, 0.1191920, 0.9503041]
#     ])
#     xyz = rgb_linear @ M_sRGB_D65.T
#     X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

#     Cx, Cz = 1.2769, 1.0592
#     YI_raw = 100 * (Cx * X - Cz * Z) / np.clip(Y, 1e-6, None)
#     YI_norm = (YI_raw - YI_raw.min()) / (YI_raw.max() - YI_raw.min() + 1e-6)
#     return YI_norm[0]

# if submit:
#     try:
#         gender_val = 1.0 if gender == "Male" else 0.0
#         yi = compute_yellowness_index(r, g, b, c)

#         input_df = pd.DataFrame([{
#             "Age": age,
#             "Gender": gender_val,
#             "BodyTemp": body_temp,
#             "LiverTemp": liver_temp,
#             "GSR": gsr,
#             "BMI": bmi,
#             "Yellowness Index": yi
#         }])

#         # Scale input
#         input_scaled = scaler.transform(input_df)

#         # Predict
#         pred = stacked_model.predict(input_scaled)[0]
#         result = "🟢 Healthy" if pred == 0 else "🔴 Unhealthy"

#         st.subheader("Prediction Result")
#         st.success(result)

#     except Exception as e:
#         st.error(f"Prediction failed: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and scaler
stacked_model = joblib.load("stacked_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="LiverGuard - Health Predictor", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    /* Background gradient or image */
    body {
        background: linear-gradient(to right, #f7f8fa, #e3f2fd);
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-top: 0px;
    }
    .form-container {
        background-color: #ffffffdd;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .result-box {
        background-color: #f1f8e9;
        color: #33691e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    .liver-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 120px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Optional Liver image on top
st.markdown('<img src="https://cdn-icons-png.flaticon.com/512/3794/3794666.png" class="liver-image">', unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">LiverGuard 🧬</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Liver Health Prediction using Sensor Data</div>', unsafe_allow_html=True)

# --- Input Form ---
with st.form("sensor_form"):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
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
    submit = st.form_submit_button("🔍 Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Yellowness Index Function ---
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

# --- Prediction Logic ---
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

        # Display result
        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")




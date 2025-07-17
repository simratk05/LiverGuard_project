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

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from pathlib import Path

# # ------------------------------------------------------------------
# # 1. Utilities
# # ------------------------------------------------------------------
# @st.cache_resource  # keeps models in memory across sessions
# def load_artifacts():
#     base = Path(_file_).with_suffix("")
#     stacked_model = joblib.load(base / "stacked_model.pkl")
#     scaler = joblib.load(base / "scaler.pkl")
#     return stacked_model, scaler

# def compute_yellowness_index(r, g, b, c):
#     rgb = np.array([[r, g, b]], dtype=float)
#     c_arr = np.array([[max(c, 1e-6)]])
#     rgb_norm = rgb / c_arr

#     gray_world = np.mean(rgb_norm, axis=0)
#     rgb_balance = np.clip(rgb_norm / (gray_world + 1e-6), 0, 1)

#     gamma = 2.2
#     rgb_lin = np.power(rgb_balance, gamma)
#     mat = np.array([[0.4124564, 0.3575761, 0.1804375],
#                     [0.2126729, 0.7151522, 0.0721750],
#                     [0.0193339, 0.1191920, 0.9503041]])
#     xyz = rgb_lin @ mat.T
#     X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

#     Cx, Cz = 1.2769, 1.0592
#     yi_raw = 100 * (Cx * X - Cz * Z) / max(Y, 1e-6)
#     return float(yi_raw)

# def predict_health(df_scaled, model):
#     pred = model.predict(df_scaled)[0]
#     status = "Healthy" if pred == 0 else "Unhealthy"
#     emoji = "🟢" if pred == 0 else "🔴"
#     return pred, f"{emoji} {status}"

# def personalised_tips(prediction):
#     """Return a list of precautionary or maintenance tips based on prediction."""
#     healthy_tips = [
#         "Continue balanced, antioxidant-rich meals (e.g., vegetables, fruit, legumes).",
#         "Limit alcohol to ≤1 standard drink/day for women or ≤2 for men.",
#         "Exercise ≥150 minutes/week (moderate) plus 2 resistance sessions."
#     ]
#     unhealthy_tips = [
#         "Schedule a medical review for comprehensive liver function testing.",
#         "Adopt a Mediterranean-style diet rich in monounsaturated fats.",
#         "Eliminate alcohol and limit ultra-processed foods with excess fructose.",
#         "Maintain BMI <25; aim for 5–10% gradual weight loss if overweight.",
#         "Verify all prescription drugs with a clinician for hepatotoxicity risk.",
#         "Ensure hepatitis A and B vaccination is up to date."
#     ]
#     return healthy_tips if prediction == 0 else unhealthy_tips

# # ------------------------------------------------------------------
# # 2. Page Config & Sidebar
# # ------------------------------------------------------------------
# st.set_page_config(page_title="LiverGuard – Liver Health Predictor", layout="centered", page_icon="🧬")
# st.sidebar.title("ℹ About LiverGuard")
# st.sidebar.markdown(
#     """
#     *LiverGuard* uses a stacked ensemble of gradient-boosting and neural-network models to classify
#     real-time sensor inputs as Healthy or Unhealthy.  
#     The algorithm was trained on anonymised clinical data (n ≈ 8,000) spanning wide age, BMI and ethnic groups.
#     Disclaimer: This tool is not a substitute for professional medical advice.  
#     """
# )

# # ------------------------------------------------------------------
# # 3. Input Form
# # ------------------------------------------------------------------
# st.title("🧬 Liver Health Prediction")

# with st.form("sensor_form", clear_on_submit=False):
#     col1, col2 = st.columns(2)
#     with col1:
#         age = st.number_input("Age (years)", 1, 100, 30)
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         bmi = st.number_input("Body-Mass Index (BMI)", 10.0, 60.0, step=0.1)
#         body_temp = st.number_input("Body Temperature (°C)", 34.0, 42.0, step=0.1)
#         liver_temp = st.number_input("Liver Temperature (°C)", 34.0, 45.0, step=0.1)
#     with col2:
#         st.markdown("##### Skin-Colour Sensor")
#         r = st.number_input("Red (R)", 0.0, 1_024.0, step=1.0)
#         g = st.number_input("Green (G)", 0.0, 1_024.0, step=1.0)
#         b = st.number_input("Blue (B)", 0.0, 1_024.0, step=1.0)
#         c = st.number_input("Intensity (C)", 0.0, 1_024.0, step=1.0)
#         gsr = st.number_input("Galvanic Skin Response (μS)", 0.0, 100.0, step=0.1)

#     submit = st.form_submit_button("▶ Predict")

# # ------------------------------------------------------------------
# # 4. Inference & Output
# # ------------------------------------------------------------------
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

#         model, scaler = load_artifacts()
#         input_scaled = scaler.transform(input_df)
#         pred_label, result_badge = predict_health(input_scaled, model)

#         # Tabs for structured results
#         tab_pred, tab_reco = st.tabs(["Prediction", "Recommendations"])

#         with tab_pred:
#             st.metric(label="Liver Health Status", value=result_badge)
#             st.caption(f"Computed Yellowness Index: *{yi:.2f}*")

#         with tab_reco:
#             st.subheader("Action Plan")
#             for tip in personalised_tips(pred_label):
#                 st.write(f"• {tip}")

#         st.toast("Inference complete", icon="✅")
#     except Exception as e:
#         st.error("Prediction failed. Details logged for review.")
#         st.exception(e)

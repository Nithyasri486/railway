import os
import sys
import streamlit as st
import pandas as pd
import joblib
import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Add parent directory to path for Firebase import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import Firebase manager
try:
    from firebase_config import firebase_manager
    # Ensure it's initialized
    firebase_manager.initialize()
    FIREBASE_ENABLED = firebase_manager.initialized
except ImportError:
    print("⚠️ Firebase config not found")
    FIREBASE_ENABLED = False
    firebase_manager = None

# ==============================
# PATH FIX (IMPORTANT)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# GROQ LLM CONFIG
# ==============================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key="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"
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a railway safety decision engine.
Analyze sensor data and ML output.
Provide:
1. Severity level
2. Risk explanation
3. Immediate actions
4. Whether to alert control room"""),
    ("human", "{data}")
])

chain = prompt | llm


def get_decision(sensor_data, prediction):
    payload = {
        "sensor_data": sensor_data,
        "ml_prediction": "FAULT" if prediction == 1 else "SAFE"
    }
    response = chain.invoke({
        "data": json.dumps(payload, indent=2)
    })
    return response.content


# ==============================
# RULE-BASED SAFETY CHECK
# ==============================
def rule_based_check(vibration, temperature, speed, noise):
    if vibration <= 0.4 and temperature <= 70 and speed <= 100 and noise <= 75:
        return "SAFE"
    if vibration >= 0.7 or temperature >= 90 or speed >= 130 or noise >= 90:
        return "FAULT"
    return "ML"


# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="SmartRail Shield", layout="centered")
st.title("🚆 SmartRail Shield – Machine Fault Detection")

st.markdown(
    "This module combines **Rule-Based Safety**, **Machine Learning Prediction**, "
    "and **AI Decision Support** using simulated sensor inputs."
)

FEATURES = [
    "vibration",
    "temperature",
    "speed",
    "noise_level",
    "pilot_drowsy",
    "obstacle_detected"
]

model_choice = st.selectbox(
    "Select Machine Learning Model",
    ["Random Forest", "XGBoost"]
)

vibration = st.slider("Vibration Level", 0.0, 1.0, 0.4)
temperature = st.slider("Engine Temperature (°C)", 20.0, 120.0, 65.0)
speed = st.slider("Train Speed (km/h)", 0, 180, 80)
noise = st.slider("Noise Level (dB)", 0, 120, 75)

sensor_input = [
    vibration,
    temperature,
    speed,
    noise,
    0,
    0
]


def load_model(model_name):
    if model_name == "Random Forest":
        model_path = os.path.join(BASE_DIR, "rf_model.pkl")
        return joblib.load(model_path)
    else:
        model_path = os.path.join(BASE_DIR, "xgb_model.pkl")
        return joblib.load(model_path)


if st.button("Predict & Analyze"):

    rule_result = rule_based_check(vibration, temperature, speed, noise)

    model = load_model(model_choice)
    df = pd.DataFrame([sensor_input], columns=FEATURES)
    ml_prediction = model.predict(df)[0]

    st.subheader("📌 Fault Status")

    if rule_result == "SAFE":
        st.success("✅ SYSTEM SAFE (Rule-Based Threshold)")
        final_prediction = 0

    elif rule_result == "FAULT":
        st.error("🚨 FAULT DETECTED (Critical Threshold Exceeded)")
        final_prediction = 1

    else:
        if ml_prediction == 1:
            st.error("🚨 FAULT DETECTED (ML Prediction)")
        else:
            st.success("✅ SYSTEM SAFE (ML Prediction)")
        final_prediction = ml_prediction

    st.subheader("🤖 AI Safety Decision Engine")

    decision = get_decision(
        dict(zip(FEATURES, sensor_input)),
        final_prediction
    )
    st.write(decision)
    
    # Save to Firebase and Send Email Alert
    if FIREBASE_ENABLED and firebase_manager:
        try:
            status_text = 'FAULT' if final_prediction == 1 else 'SAFE'
            fault_data = {
                'module_name': 'Machine Fault Detection',
                'model': model_choice,
                'vibration': vibration,
                'temperature': temperature,
                'speed': speed,
                'noise': noise,
                'prediction': status_text,
                'ai_decision': decision[:300]
            }
            # Central PUSH to Firebase
            firebase_manager.push_to_realtime('fault_detections', fault_data)
            st.success("✅ Fault Data Pushed to Central Firebase")

            # Send Email Alert if Fault is detected
            if final_prediction == 1:
                from email_utils import send_email_alert
                subject = "🚨 CRITICAL: Machine Fault Detected - SmartRail Shield"
                body = f"""
                Attention Control Room,

                A critical machine fault has been detected by the SmartRail Shield system.

                --- Fault Details ---
                Detected Status: FAULT
                Vibration: {vibration}
                Temperature: {temperature}°C
                Speed: {speed} km/h
                Noise Level: {noise} dB
                Model Used: {model_choice}

                --- AI Decision Engine Analysis ---
                {decision}

                Please take immediate action.
                """
                if send_email_alert(subject, body):
                    st.info("📧 Email Alert Sent to Control Room")

        except Exception as e:
            st.warning(f"⚠️ Firebase/Email operation failed: {str(e)}")


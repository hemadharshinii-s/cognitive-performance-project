import streamlit as st
import joblib
from pipeline import predict_dsst, predict_cognitive_risk, train_nhanes_model, train_brfss_model

st.set_page_config(page_title="Cognitive Health Dashboard", layout="centered")

st.title("🧠 Cognitive Health & Environment Analysis")

# =========================
# LOAD MODELS
# =========================

try:
    nhanes_model = joblib.load("models/nhanes_model.pkl")
    brfss_model = joblib.load("models/brfss_model.pkl")
except:
    st.warning("Training models...")
    nhanes_model, _ = train_nhanes_model()
    brfss_model, _ = train_brfss_model()


# =========================
# TAB 1: NHANES PREDICTION
# =========================

st.header("📊 Predict Cognitive Performance (DSST)")

age = st.slider("Age", 60, 80, 70)
gender = st.selectbox("Gender", ["male","female"])
education = st.selectbox("Education", ["<9th","9-11th","HS","some_college","college"])
bmi = st.slider("BMI", 15.0, 45.0, 28.0)
smoking = st.selectbox("Smoking", ["smoker","non_smoker"])
activity = st.selectbox("Physical Activity", ["active","inactive"])

if st.button("Predict DSST Score"):
    score = predict_dsst(nhanes_model, age, gender, education, bmi, smoking, activity)
    st.success(f"Predicted DSST Score: {score:.2f}")


# =========================
# TAB 2: BRFSS RISK MODEL
# =========================

st.header("🌫️ Cognitive Risk (Environmental Model)")

aqi = st.slider("AQI", 10, 200, 50)
age2 = st.slider("Age Group", 1, 14, 10)
edu2 = st.slider("Education Level", 1, 6, 4)
bmi2 = st.slider("BMI (scaled)", 1000, 5000, 2500)
smoke2 = st.selectbox("Ever Smoked 100 Cigarettes", [1,2])
diabetes = st.selectbox("Diabetes Status", [1,2,3])

if st.button("Predict Cognitive Risk"):
    result = predict_cognitive_risk(brfss_model, aqi, age2, edu2, bmi2, smoke2, diabetes)

    if result == 1:
        st.error("Higher risk of cognitive impairment")
    else:
        st.success("Lower risk of cognitive impairment")


# =========================
# FOOTER INSIGHT
# =========================

st.markdown("---")
st.caption("Models based on NHANES + BRFSS + EPA AQI datasets (2011–2022)")
import streamlit as st
import numpy as np
import pickle
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wine Quality Prediction 🍷",
    layout="wide"
)

# ---------------- CSS (BACKGROUND + PREMIUM SLIDER) ----------------
st.markdown("""
<style>
html, body, .stApp {
    height: 100vh;
    overflow: hidden !important;
}

/* Background image with dark overlay */
.stApp {
    background:
        linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
        url("https://img.freepik.com/free-photo/side-view-red-wine-bottle-glass-grape-dark-table-horizontal_176474-4123.jpg?semt=ais_hybrid&w=740&q=80");
    background-size: cover;
    background-position: center;
}

/* Reduce default padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

/* Glass panel */
.glass {
    background: rgba(0,0,0,0.55);
    padding: 18px;
    border-radius: 16px;
    color: #F5E6C8;
}

/* Headings */
h1, h2 {
    color: #F5E6C8 !important;
    text-align: center;
}

/* Sub text */
p {
    color: #E6D3A3 !important;
    text-align: center;
}

/* ================= COOL PREMIUM SLIDER ================= */

/* Base track */
div[data-baseweb="slider"] > div > div {
    height: 6px !important;
    background: rgba(255, 255, 255, 0.18) !important;
    border-radius: 10px;
}

/* Filled track (wine gradient) */
div[data-baseweb="slider"] div div div {
    height: 6px !important;
    background: linear-gradient(
        90deg,
        #5A0F1B,
        #7B1E1E,
        #A83232
    ) !important;
    border-radius: 10px;
}

/* Thumb (glassy gold knob) */
div[data-baseweb="slider"] div[role="slider"] {
    width: 16px !important;
    height: 16px !important;
    background: radial-gradient(
        circle,
        #FFF3C4 30%,
        #E6B566 60%,
        #B8860B 100%
    ) !important;
    border: none !important;
    box-shadow: 0 0 8px rgba(255, 215, 0, 0.6) !important;
}

/* Labels */
label {
    color: #F5E6C8 !important;
    font-weight: 500;
}

/* Value bubble */
.stSlider span {
    background: rgba(0, 0, 0, 0.55);
    color: #F5E6C8 !important;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 12px;
}

/* Reduce slider spacing */
div[data-baseweb="slider"] {
    margin-top: -10px;
}

/* Button */
.stButton > button {
    background-color: #7B1E1E;
    color: #F5E6C8;
    border-radius: 14px;
    border: none;
    padding: 0.6rem 1.4rem;
    font-size: 16px;
}

/* Alerts */
.stAlert-success {
    background-color: rgba(124, 252, 152, 0.15);
    color: #7CFC98;
}
.stAlert-info {
    background-color: rgba(255, 215, 0, 0.15);
    color: #FFD700;
}
.stAlert-warning {
    background-color: rgba(255, 111, 97, 0.18);
    color: #FF6F61;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<div class="glass">
    <h2>🍷 Wine Quality Prediction 🍷</h2>
    <p>Predict wine quality using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("finalized_model.sav", "rb"))
scaler = pickle.load(open("scaler_model.sav", "rb"))

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.0)
    chlorides = st.slider("Chlorides", 0.01, 0.6, 0.08)

with col2:
    free_sulfur = st.slider("Free Sulfur Dioxide", 1.0, 70.0, 15.0)
    total_sulfur = st.slider("Total Sulfur Dioxide", 6.0, 300.0, 46.0)
    density = st.slider("Density", 0.990, 1.005, 0.996)
    pH = st.slider("pH", 2.8, 4.0, 3.3)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6)

alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.0)

# ---------------- BUTTON ----------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
predict = st.button("🍷 Predict Wine Quality")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RESULT + GIF + AUDIO ----------------
if predict:
    with st.spinner("Analyzing wine quality..."):
        time.sleep(1)

    user_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        np.log(residual_sugar + 1),
        np.log(chlorides + 1),
        np.log(free_sulfur + 1),
        np.log(total_sulfur + 1),
        density,
        pH,
        np.log(sulphates + 1),
        alcohol
    ]])

    user_data = scaler.transform(user_data)
    prediction = model.predict(user_data)[0]

    st.success(f"🎯 Predicted Wine Quality: {int(prediction)}")

    if prediction >= 7:
        st.info("🍷 Excellent Wine Quality")
    elif prediction >= 5:
        st.info("🙂 Average Wine Quality")
    else:
        st.warning("⚠️ Low Wine Quality")

    # Wine pouring GIF
    st.image(
        "https://i.pinimg.com/originals/e9/ce/ca/e9ceca2e61ea591b4986e997f9a6b72d.gif",
        width=260
    )

    # Local audio (user clicks play – browser rule)
    st.audio("wine_pour.mp3", format="audio/mp3")


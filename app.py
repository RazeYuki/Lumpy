import streamlit as st
import numpy as np
import pickle

# ----------------------------------
# Load Model & Scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------------
# Judul Aplikasi
st.title("Prediksi Lumpy Skin Disease (LSD) pada Ternak")

st.markdown("Silakan masukkan nilai dari fitur-fitur berikut untuk memprediksi apakah ternak berisiko terkena penyakit LSD atau tidak.")

# ----------------------------------
# Input Form
x = st.number_input("Longitude (x)", value=0.0)
y = st.number_input("Latitude (y)", value=0.0)
X5_Ct_2010_Da = st.number_input("Kepadatan Sapi per Km² (2010)", value=0.0)
vap = st.number_input("Tekanan Uap Air (Vapor Pressure)", value=0.0)
tmn = st.number_input("Temperatur Minimum Harian (°C)", value=0.0)

# Submit Button
if st.button("Prediksi"):
    # Data input sebagai array
    input_data = np.array([[x, y, X5_Ct_2010_Da, vap, tmn]])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

    # Output
    if prediction == 1:
        st.error("🚨 Hasil Prediksi: POSITIF terkena LSD")
    else:
        st.success("✅ Hasil Prediksi: NEGATIF dari LSD")

    # Tampilkan probabilitas jika ada
    if proba is not None:
        st.write(f"Probabilitas Positif: {proba[1]*100:.2f}%")
        st.write(f"Probabilitas Negatif: {proba[0]*100:.2f}%")

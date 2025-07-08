import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------
# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------
# Judul Aplikasi
st.set_page_config(page_title="Prediksi LSD", layout="centered")
st.title("🐄 Prediksi Lumpy Skin Disease (LSD) pada Ternak")
st.markdown("Masukkan data lingkungan & lokasi untuk memprediksi apakah ternak berisiko terkena LSD atau tidak.")

# ---------------------------------------
# Form Input
st.subheader("📝 Input Fitur Penting")

x = st.number_input("Longitude (x)", value=110.0, format="%.6f")
y = st.number_input("Latitude (y)", value=-7.0, format="%.6f")
X5_Ct_2010_Da = st.number_input("Kepadatan Sapi per Km² (2010)", value=50.0)
vap = st.number_input("Tekanan Uap Air (Vapor Pressure)", value=15.0)
tmn = st.number_input("Temperatur Minimum Harian (°C)", value=20.0)

if st.button("🔍 Prediksi LSD"):
    input_data = np.array([[x, y, X5_Ct_2010_Da, vap, tmn]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Jika model punya probabilitas
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        st.write(f"🔵 Probabilitas Negatif: {proba[0]*100:.2f}%")
        st.write(f"🔴 Probabilitas Positif: {proba[1]*100:.2f}%")

    if prediction == 1:
        st.error("🚨 Hasil Prediksi: POSITIF terkena LSD")
    else:
        st.success("✅ Hasil Prediksi: NEGATIF dari LSD")

# ---------------------------------------
# Visualisasi Peta
st.subheader("🗺️ Peta Persebaran Kasus LSD")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/RazeYuki/Lumpy/main/Lumpy%20skin%20disease%20data.csv"
    df = pd.read_csv(url)
    
    from sklearn.preprocessing import LabelEncoder
    if df["lumpy"].dtype == "object":
        df["lumpy"] = LabelEncoder().fit_transform(df["lumpy"])

    df = df.drop_duplicates()
    df = df.dropna(subset=["x", "y"])
    return df

data = load_data()

map_data = data[["x", "y", "lumpy"]].copy()
map_data.rename(columns={"x": "lon", "y": "lat"}, inplace=True)
map_data["label"] = map_data["lumpy"].map({1: "Positif LSD", 0: "Negatif"})

selected_label = st.selectbox("🎯 Filter Label:", ["Semua", "Positif LSD", "Negatif"])
if selected_label != "Semua":
    map_data = map_data[map_data["label"] == selected_label]

st.map(map_data[["lat", "lon"]])

# ---------------------------------------
st.markdown("---")
st.caption("Dibuat untuk keperluan prediksi penyakit ternak berbasis Machine Learning dan Streamlit.")

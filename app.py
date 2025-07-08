import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------
# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------
st.set_page_config(page_title="Prediksi LSD", layout="centered")
st.title("🐄 Prediksi Penyakit Lumpy Skin Disease (LSD) pada Ternak")
st.markdown("Gunakan data lokasi & lingkungan untuk memprediksi kemungkinan terkena LSD.")

# ---------------------------------------
@st.cache_data
def load_dataset():
    url = "https://raw.githubusercontent.com/RazeYuki/Lumpy/main/Lumpy%20skin%20disease%20data.csv"
    df = pd.read_csv(url)
    df = df.drop_duplicates()
    df = df.dropna(subset=["region", "x", "y"])
    return df

@st.cache_data
def get_lokasi_options():
    df = load_dataset()
    lokasi_map = df[["region", "x", "y"]].drop_duplicates().set_index("region").to_dict(orient="index")
    return lokasi_map

# ---------------------------------------
# Input Lokasi
st.subheader("📍 Pilih Lokasi Peternakan")
lokasi_dict = get_lokasi_options()
selected_region = st.selectbox("Pilih wilayah:", list(lokasi_dict.keys()))
x = lokasi_dict[selected_region]["x"]
y = lokasi_dict[selected_region]["y"]
st.write(f"Koordinat otomatis: **Longitude = {x:.4f}**, **Latitude = {y:.4f}**")

# Input Fitur Lingkungan
st.subheader("🧪 Data Lingkungan")
X5_Ct_2010_Da = st.number_input("🐄 Kepadatan Populasi Sapi per km² ", value=50.0)
vap = st.number_input("💧 Tekanan Uap Air (Vapor Pressure)", value=15.0)
tmn = st.number_input("🌡️ Suhu Minimum Harian (°C)", value=20.0)

# ---------------------------------------
# Prediksi
if st.button("🔍 Prediksi LSD"):
    input_data = np.array([[x, y, X5_Ct_2010_Da, vap, tmn]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        st.info(f"🔵 Probabilitas Negatif: {proba[0]*100:.2f}%")
        st.warning(f"🔴 Probabilitas Positif: {proba[1]*100:.2f}%")

    if prediction == 1:
        st.error("🚨 Hasil Prediksi: POSITIF terkena LSD")
    else:
        st.success("✅ Hasil Prediksi: NEGATIF dari LSD")

# ---------------------------------------
# Peta Persebaran
st.subheader("🗺️ Peta Persebaran Kasus LSD")

data = load_dataset()
from sklearn.preprocessing import LabelEncoder
if data["lumpy"].dtype == "object":
    data["lumpy"] = LabelEncoder().fit_transform(data["lumpy"])

map_data = data[["x", "y", "lumpy"]].copy()
map_data.rename(columns={"x": "lon", "y": "lat"}, inplace=True)
map_data["label"] = map_data["lumpy"].map({1: "Positif LSD", 0: "Negatif"})

filter_label = st.selectbox("🎯 Tampilkan kasus:", ["Semua", "Positif LSD", "Negatif"])
if filter_label != "Semua":
    map_data = map_data[map_data["label"] == filter_label]

st.map(map_data[["lat", "lon"]])

# ---------------------------------------
st.markdown("---")
st.caption("🚧 Dibuat untuk demonstrasi prediksi penyakit ternak berbasis Machine Learning dan lokasi spasial.")

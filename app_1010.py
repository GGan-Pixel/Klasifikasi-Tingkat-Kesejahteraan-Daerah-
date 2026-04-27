import streamlit as st
import joblib
import numpy as np
import pandas as pd

# D1 — Muat model, scaler, dan encoder
model       = joblib.load('model_terbaik.pkl')
scaler      = joblib.load('scaler.pkl')
le_provinsi = joblib.load('le_provinsi.pkl')
le_kabkota  = joblib.load('le_kabkota.pkl')
le_kategori = joblib.load('le_kategori.pkl')

# D4 — Header aplikasi
st.title(" Prediksi Tingkat Kesejahteraan Daerah")
st.write("Masukkan data daerah untuk memprediksi tingkat kesejahteraan.")

# D3 — Widget input
col1, col2 = st.columns(2)

with col1:
    provinsi     = st.selectbox("Provinsi", le_provinsi.classes_)
    kabkota      = st.selectbox("Kabupaten / Kota", le_kabkota.classes_)
    kategori     = st.selectbox("Kategori Wilayah", le_kategori.classes_)
    jumlah_pddk  = st.number_input("Jumlah Penduduk", min_value=1000, value=500000, step=1000)
    rata_sekolah = st.slider("Rata-rata Lama Sekolah (tahun)", 4.0, 14.0, 8.5)

with col2:
    aps          = st.slider("Angka Partisipasi Sekolah (%)", 50.0, 100.0, 92.3)
    pengangguran = st.slider("Angka Pengangguran (%)", 1.0, 25.0, 7.8)
    rumah_layak  = st.slider("Persentase Rumah Layak (%)", 30.0, 100.0, 78.5)
    air_bersih   = st.slider("Akses Air Bersih (%)", 30.0, 100.0, 85.0)
    pdrb         = st.number_input("PDRB per Kapita (juta rupiah)", min_value=5.0, max_value=200.0, value=45.2)

# D2 — Fungsi prediksi
def predict(inputs):
    arr = np.array(inputs).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    prob = model.predict_proba(arr_scaled).max() * 100
    return pred, prob

# Tombol prediksi
if st.button("Prediksi Sekarang"):

    # Encode input kategorikal — harus pakai encoder yang sama dari notebook!
    provinsi_enc = le_provinsi.transform([provinsi])[0]
    kabkota_enc  = le_kabkota.transform([kabkota])[0]
    kategori_enc = le_kategori.transform([kategori])[0]

    # Susun input sesuai urutan kolom waktu training:
    # Provinsi, Kabupaten_Kota, Jumlah_Penduduk, Rata_Lama_Sekolah,
    # Angka_Partisipasi_Sekolah, Angka_Pengangguran,
    # Persentase_Rumah_Layak, Akses_Air_Bersih, PDRB_per_Kapita, Kategori_Wilayah
    inputs = [
        provinsi_enc, kabkota_enc, jumlah_pddk,
        rata_sekolah, aps, pengangguran,
        rumah_layak, air_bersih, pdrb,
        kategori_enc
    ]

    hasil, keyakinan = predict(inputs)

    # Tampilkan hasil
    st.divider()
    if hasil == "Tinggi":
        st.success(f"###  Tingkat Kesejahteraan: {hasil}")
    elif hasil == "Sedang":
        st.warning(f"###  Tingkat Kesejahteraan: {hasil}")
    else:
        st.error(f"###  Tingkat Kesejahteraan: {hasil}")

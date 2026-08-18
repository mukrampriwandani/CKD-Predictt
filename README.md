# Sistem Prediksi Penyakit Ginjal Kronis

Aplikasi Streamlit untuk prediksi CKD / Not CKD menggunakan Random Forest dan dataset `kidney_disease.csv`.
Proses training model dipisahkan ke `train_model.ipynb`, sedangkan aplikasi berada di `app.py`.

## Fitur

- Input data pasien
- Hasil prediksi CKD / Not CKD
- Confidence, tingkat risiko, dan keterangan hasil
- Dashboard jumlah data pasien, CKD, non-CKD, dan grafik hasil prediksi
- Riwayat prediksi yang tersimpan ke `prediction_history.csv`
- Upload CSV dataset penyakit ginjal kronis
- Confusion matrix
- Grafik akurasi Random Forest
- Feature importance Random Forest

## Struktur File

- `train_model.ipynb`: training, evaluasi, confusion matrix, grafik akurasi, feature importance, dan penyimpanan model.
- `model_artifacts.joblib`: hasil training yang dipakai aplikasi.
- `app.py`: aplikasi Streamlit untuk input pasien, prediksi, dashboard, dan riwayat.
- `kidney_disease.csv`: dataset bawaan.

## Jalankan Lokal

```bash
pip install -r requirements.txt
jupyter notebook train_model.ipynb
streamlit run app.py
```

Jalankan semua cell di `train_model.ipynb` terlebih dahulu agar file `model_artifacts.joblib` dibuat. Setelah itu baru jalankan `app.py`.

## Deployment Streamlit Community Cloud

1. Upload semua file ini ke repository GitHub:
   - `app.py`
   - `train_model.ipynb`
   - `model_artifacts.joblib`
   - `kidney_disease.csv`
   - `requirements.txt`
   - `.streamlit/config.toml`
2. Buka `https://share.streamlit.io/`.
3. Pilih repository, branch, dan main file `app.py`.
4. Klik Deploy.

Catatan: `prediction_history.csv` bisa dibuat otomatis saat ada prediksi baru. Pada hosting gratis, file riwayat dapat hilang saat aplikasi restart.

# Klasifikasi Penyakit Jantung (Decision Tree + Flask)

UTS Kecerdasan Buatan 2026, topik **Klasifikasi Data**.

## Struktur
```
app.py            aplikasi Flask (4 halaman)
train.py          preprocessing, split, training, evaluasi, simpan model
data/heart.csv    dataset Cleveland (303 baris, 13 fitur + target)
model/            decision_tree.joblib + metrics.json (hasil train.py)
static/           style.css + gambar hasil (confusion matrix, pohon, importance)
templates/        halaman HTML
```

## Jalankan lokal
```
pip install -r requirements.txt
python train.py      # melatih ulang model (opsional, hasil sudah disertakan)
python app.py        # buka http://127.0.0.1:5000
```

## Deploy (agar bisa diakses online)
**Render** (gratis): push folder ini ke GitHub → New Web Service → pilih repo →
Build command `pip install -r requirements.txt`, Start command `gunicorn app:app`.

**PythonAnywhere** (gratis): upload folder, buat virtualenv, `pip install -r requirements.txt`,
lalu buat Web app tipe Flask dan arahkan WSGI ke `app.py` (`from app import app as application`).

Jika versi scikit-learn di server berbeda dari saat training, jalankan `python train.py`
di server (atau ubah `requirements.txt` ke versi yang sama) agar file model terbaca.

## Hal yang perlu kamu cek sebelum presentasi
1. **Jurnal acuan.** Dataset dan akurasi 93,44% diambil dari ringkasan artikel JATI.
   Buka PDF-nya, isi `precision`/`recall` di dict `JURNAL` pada `app.py` kalau ada, dan
   sesuaikan paragraf perbandingan di `templates/perbandingan.html` dengan isi artikel.
2. **Kode kategori.** Salinan CSV ini memakai kode berbeda dari berkas asli UCI. Dari data:
   `target` 0 = sakit dan 1 = sehat (dibalik di `train.py`), `cp` 0 = tanpa gejala dan 3 = angina
   tipikal, `slope` 0 = menurun dan 2 = menanjak. Label form mengikuti pola ini.
3. **Sumber dataset.** Sebutkan UCI (Cleveland) sebagai sumber asli, sesuai syarat soal.

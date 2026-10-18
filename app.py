"""Aplikasi web Flask: klasifikasi penyakit jantung dengan Decision Tree."""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).parent
app = Flask(__name__)

MODEL = joblib.load(BASE / "model" / "decision_tree.joblib")
METRICS = json.loads((BASE / "model" / "metrics.json").read_text())
DATASET = pd.read_csv(BASE / "data" / "heart.csv")

# --------------------------------------------------------------------------
# Definisi fitur untuk form. Urutan HARUS sama dengan METRICS["fitur"].
# Kode kategori mengikuti file heart.csv yang dipakai (lihat README).
# --------------------------------------------------------------------------
FIELDS = [
    {"name": "age", "label": "Usia", "unit": "tahun", "type": "number",
     "min": 20, "max": 90, "step": 1, "default": 54},
    {"name": "sex", "label": "Jenis kelamin", "type": "select",
     "options": [(1, "Laki-laki"), (0, "Perempuan")], "default": 1},
    {"name": "cp", "label": "Tipe nyeri dada", "type": "select",
     "options": [(3, "Angina tipikal"), (2, "Angina atipikal"),
                 (1, "Nyeri non-angina"), (0, "Tanpa gejala")], "default": 0},
    {"name": "trestbps", "label": "Tekanan darah istirahat", "unit": "mm Hg",
     "type": "number", "min": 80, "max": 220, "step": 1, "default": 130},
    {"name": "chol", "label": "Kolesterol serum", "unit": "mg/dl",
     "type": "number", "min": 100, "max": 600, "step": 1, "default": 245},
    {"name": "fbs", "label": "Gula darah puasa > 120 mg/dl", "type": "select",
     "options": [(0, "Tidak"), (1, "Ya")], "default": 0},
    {"name": "restecg", "label": "Hasil EKG saat istirahat", "type": "select",
     "options": [(0, "Normal"), (1, "Kelainan gelombang ST-T"),
                 (2, "Hipertrofi ventrikel kiri")], "default": 1},
    {"name": "thalach", "label": "Detak jantung maksimum", "unit": "bpm",
     "type": "number", "min": 60, "max": 220, "step": 1, "default": 150},
    {"name": "exang", "label": "Angina saat olahraga", "type": "select",
     "options": [(0, "Tidak"), (1, "Ya")], "default": 0},
    {"name": "oldpeak", "label": "ST depression (oldpeak)", "type": "number",
     "min": 0, "max": 7, "step": 0.1, "default": 1.0},
    {"name": "slope", "label": "Kemiringan segmen ST", "type": "select",
     "options": [(2, "Menanjak"), (1, "Datar"), (0, "Menurun")], "default": 1},
    {"name": "ca", "label": "Jumlah pembuluh darah utama (0-3)",
     "type": "select", "options": [(0, "0"), (1, "1"), (2, "2"), (3, "3")],
     "default": 0},
    {"name": "thal", "label": "Hasil thalassemia", "type": "select",
     "options": [(2, "Normal"), (1, "Cacat tetap"), (3, "Cacat reversibel")],
     "default": 2},
]
assert [f["name"] for f in FIELDS] == METRICS["fitur"], "Urutan fitur tidak cocok"

# Hasil artikel jurnal acuan -> isi/ubah sesuai artikel yang kamu pilih.
JURNAL = {
    "judul": "Penerapan Algoritma Decision Tree Berbasis Pohon Keputusan "
             "dalam Klasifikasi Penyakit Jantung",
    "penerbit": "JATI (Jurnal Mahasiswa Teknik Informatika)",
    "url": "https://ejournal.itn.ac.id/index.php/jati/article/download/8264/4886",
    "dataset": "Cleveland Clinic Foundation (UCI Machine Learning Repository)",
    "accuracy": 0.9344,
    # Isi dari artikel jika tersedia, atau biarkan None.
    "precision": None,
    "recall": None,
}


def parse_form(form):
    """Ubah input form menjadi satu baris DataFrame; kembalikan (df, error)."""
    values = {}
    for f in FIELDS:
        raw = (form.get(f["name"]) or "").strip().replace(",", ".")
        try:
            v = float(raw)
        except ValueError:
            return None, f"Isi kolom “{f['label']}” dengan angka."
        if f["type"] == "number" and not f["min"] <= v <= f["max"]:
            return None, (f"{f['label']} harus antara {f['min']} dan "
                          f"{f['max']}.")
        if f["type"] == "select" and v not in [o[0] for o in f["options"]]:
            return None, f"Pilihan pada “{f['label']}” tidak valid."
        values[f["name"]] = v
    return pd.DataFrame([values], columns=METRICS["fitur"]), None


@app.route("/")
def index():
    info = METRICS["info_data"]
    return render_template(
        "index.html",
        info=info,
        sample=DATASET.head(8).to_dict("records"),
        columns=list(DATASET.columns),
        page="index",
    )


@app.route("/model")
def model_page():
    return render_template("model.html", m=METRICS, page="model")


@app.route("/uji", methods=["GET", "POST"])
def uji():
    result, error = None, None
    form = request.form if request.method == "POST" else {}
    if request.method == "POST":
        row, error = parse_form(request.form)
        if row is not None:
            proba = MODEL.predict_proba(row)[0]
            label = int(MODEL.predict(row)[0])
            result = {
                "label": label,
                "prob_sakit": round(float(proba[1]) * 100, 1),
                "prob_sehat": round(float(proba[0]) * 100, 1),
            }
    return render_template(
        "uji.html", fields=FIELDS, form=form, result=result, error=error,
        page="uji",
    )


@app.route("/perbandingan")
def perbandingan():
    return render_template("perbandingan.html", m=METRICS, j=JURNAL,
                           page="perbandingan")


if __name__ == "__main__":
    app.run(debug=True)

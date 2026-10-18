"""Latih model Decision Tree untuk klasifikasi penyakit jantung.

Jalankan sekali sebelum menjalankan aplikasi:
    python train.py

Output (folder model/ dan static/):
    model/decision_tree.joblib, model/metrics.json
    static/confusion_matrix.png, static/decision_tree.png, static/feature_importance.png
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

BASE = Path(__file__).parent
DATA = BASE / "data" / "heart.csv"
MODEL_DIR = BASE / "model"
STATIC_DIR = BASE / "static"
MODEL_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "target"
# Catatan penting: pada file CSV ini kolom `target` BERLAWANAN dengan yang sering
# ditulis di notebook: target = 0 -> pasien PENYAKIT JANTUNG, target = 1 -> TIDAK ada
# penyakit (terbukti dari rata-rata thalach, oldpeak, exang, dan ca per kelas).
# Agar precision/recall bermakna secara medis, label dibalik menjadi:
#   1 = terdeteksi penyakit jantung (kelas positif), 0 = tidak.

# ---------------------------------------------------------------- 1. Data
df = pd.read_csv(DATA)
info = {
    "jumlah_baris_awal": int(len(df)),
    "jumlah_kolom": int(df.shape[1]),
    "missing_value": int(df.isna().sum().sum()),
    "duplikat": int(df.duplicated().sum()),
}

# ------------------------------------------------------- 2. Preprocessing
# Dataset sudah numerik, jadi cukup: hapus duplikat dan pastikan tidak ada NaN.
# Decision Tree tidak sensitif terhadap skala fitur, sehingga scaling tidak perlu.
df = df.drop_duplicates().dropna().reset_index(drop=True)
df[TARGET] = 1 - df[TARGET]  # 1 = penyakit jantung, 0 = tidak
info["jumlah_baris_akhir"] = int(len(df))
info["distribusi_kelas"] = {
    str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()
}

X = df.drop(columns=[TARGET])
y = df[TARGET]
fitur = list(X.columns)

# ------------------------------------------- 3. Pembagian train / testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
info["data_training"] = int(len(X_train))
info["data_testing"] = int(len(X_test))

# ------------------------------------------------------------ 4. Model
# Pohon dibatasi kedalamannya lewat GridSearchCV (cross validation pada data
# training saja) supaya tidak overfitting.
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6],
    "min_samples_leaf": [1, 3, 5, 10],
    "class_weight": [None, "balanced"],
}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# ---------------------------------------------------------- 5. Evaluasi
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = (int(v) for v in cm.ravel())

metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred)), 4),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "accuracy_train": round(float(model.score(X_train, y_train)), 4),
    "cv_accuracy_terbaik": round(float(grid.best_score_), 4),
    "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    "best_params": grid.best_params_,
    "fitur": fitur,
    "info_data": info,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
}

# ---------------------------------------------------------- 6. Visual
fig, ax = plt.subplots(figsize=(5, 4.2))
ConfusionMatrixDisplay(cm, display_labels=["0", "1"]).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title("Confusion matrix (data testing)")
fig.tight_layout()
fig.savefig(STATIC_DIR / "confusion_matrix.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(
    model,
    feature_names=fitur,
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
fig.tight_layout()
fig.savefig(STATIC_DIR / "decision_tree.png", dpi=110)
plt.close(fig)

imp = pd.Series(model.feature_importances_, index=fitur).sort_values()
imp = imp[imp > 0]
fig, ax = plt.subplots(figsize=(6, 4.5))
imp.plot.barh(ax=ax, color="#2f6f8f")
ax.set_xlabel("Importance")
ax.set_title("Fitur paling berpengaruh")
fig.tight_layout()
fig.savefig(STATIC_DIR / "feature_importance.png", dpi=130)
plt.close(fig)
metrics["importance"] = {k: round(float(v), 4) for k, v in imp[::-1].items()}

# ---------------------------------------------------------- 7. Simpan
joblib.dump(model, MODEL_DIR / "decision_tree.joblib")
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

print("Parameter terbaik :", grid.best_params_)
print("Accuracy  (test)  :", metrics["accuracy"])
print("Precision (test)  :", metrics["precision"])
print("Recall    (test)  :", metrics["recall"])
print("Confusion matrix  :", cm.tolist())

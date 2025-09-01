# src/choose_threshold.py
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

SEED = 42
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH  = os.path.join(ROOT, "data", "cv_train.csv")
MODEL_PATH = os.path.join(ROOT, "models", "modelo_selector_cv.joblib")

# 1) Carga datos y mismo split que en train.py
df = pd.read_csv(DATA_PATH).dropna(subset=["texto_cv","label"])
X = df["texto_cv"].astype(str)
y = df["label"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# 2) Carga modelo entrenado y obtiene probabilidades en test
pipe = joblib.load(MODEL_PATH)
probs = pipe.predict_proba(Xte)[:,1]

# 3) Curva Precision-Recall y búsqueda de umbral con objetivos:
#    precision >= 0.60 y recall >= 0.80
prec, rec, thr = precision_recall_curve(yte, probs)  # prec/rec tienen len = len(thr)+1
prec_t, rec_t, thr_t = prec[:-1], rec[:-1], thr

f1 = 2 * (prec_t * rec_t) / (prec_t + rec_t + 1e-12)
mask = (prec_t >= 0.60) & (rec_t >= 0.80)

if mask.any():
    idx = np.argmax(f1[mask])  # mejor F1 dentro de la zona objetivo
    cand_indices = np.where(mask)[0]
    best_i = cand_indices[idx]
    best_thr, best_p, best_r, best_f1 = thr_t[best_i], prec_t[best_i], rec_t[best_i], f1[best_i]
    print(f"✅ Umbral recomendado: {best_thr:.3f} | precision={best_p:.3f} recall={best_r:.3f} F1={best_f1:.3f}")
else:
    # si no hay umbral que cumpla ambos, mostramos el mejor F1 global y tres candidatos cercanos
    best_i = int(np.argmax(f1))
    best_thr, best_p, best_r, best_f1 = thr_t[best_i], prec_t[best_i], rec_t[best_i], f1[best_i]
    print("⚠️ No hay umbral que alcance precision≥0.60 y recall≥0.80 simultáneamente.")
    print(f"Sugerencia por mejor F1 global: thr={best_thr:.3f} | precision={best_p:.3f} recall={best_r:.3f} F1={best_f1:.3f}")

# 4) Tabla breve de candidatos alrededor del mejor índice
lo = max(0, best_i - 5); hi = min(len(thr_t), best_i + 6)
print("\nUmbrales cercanos:")
print("thr\tprecision\trecall\tF1")
for i in range(lo, hi):
    print(f"{thr_t[i]:.3f}\t{prec_t[i]:.3f}\t\t{rec_t[i]:.3f}\t{f1[i]:.3f}")

# 5) Métricas finales usando el umbral elegido
preds = (probs >= best_thr).astype(int)
P = precision_score(yte, preds, zero_division=0)
R = recall_score(yte, preds, zero_division=0)
F1 = f1_score(yte, preds, zero_division=0)
print(f"\nMétricas con thr={best_thr:.3f} -> precision={P:.3f} recall={R:.3f} F1={F1:.3f}")

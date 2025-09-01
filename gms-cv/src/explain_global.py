# gms-cv/src/explain_global.py
import os, joblib, numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "modelo_selector_cv.joblib")
OUT_PATH   = os.path.join(ROOT, "reports", "explicabilidad_global.csv")

pipe = joblib.load(MODEL_PATH)
vec  = pipe.named_steps["tfidf"]
clf  = pipe.named_steps["clf"]

feature_names = vec.get_feature_names_out()
coefs = clf.coef_.ravel()  # (n_features,)

# Top 25 que MÁS favorecen la clase positiva (preselección)
top_pos_idx = np.argsort(coefs)[-25:][::-1]
top_pos = pd.DataFrame({
    "token": feature_names[top_pos_idx],
    "peso":  coefs[top_pos_idx],
    "sentido": "favorece_preseleccion"
})

# Top 25 que MÁS perjudican (empujan a clase 0)
top_neg_idx = np.argsort(coefs)[:25]
top_neg = pd.DataFrame({
    "token": feature_names[top_neg_idx],
    "peso":  coefs[top_neg_idx],
    "sentido": "favorece_descarte"
})

out = pd.concat([top_pos, top_neg], ignore_index=True)
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
out.to_csv(OUT_PATH, index=False, encoding="utf-8")

print(out.head(10))
print(f"\n✅ Guardado: {OUT_PATH}")

import os, argparse, pandas as pd, joblib
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "modelo_selector_cv.joblib")

parser = argparse.ArgumentParser()
parser.add_argument("--in_csv", required=True, help="CSV de entrada con columna de texto (p. ej. data/cv_train.csv)")
parser.add_argument("--col_texto", default="texto_cv", help="Nombre de la columna de texto")
parser.add_argument("--umbral", type=float, default=0.606, help="Umbral de decisión para preselección")
parser.add_argument("--out_csv", default=os.path.join(ROOT, "reports", "predicciones_thr.csv"),
                    help="Ruta de salida para el CSV con puntuaciones")
args = parser.parse_args()

os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)

# Carga del modelo
pipe = joblib.load(MODEL_PATH)

# Lee datos
df = pd.read_csv(args.in_csv)
if args.col_texto not in df.columns:
    raise ValueError(f"No encuentro la columna '{args.col_texto}' en {args.in_csv}")

# Predicción
scores = pipe.predict_proba(df[args.col_texto].astype(str))[:, 1]
df_out = df.copy()
df_out["score_selector"] = scores
df_out["preseleccionado"] = (df_out["score_selector"] >= args.umbral).astype(int)

# Métricas si hay etiqueta disponible
if "label" in df_out.columns:
    P = precision_score(df_out["label"], df_out["preseleccionado"], zero_division=0)
    R = recall_score(df_out["label"], df_out["preseleccionado"], zero_division=0)
    F1 = f1_score(df_out["label"], df_out["preseleccionado"], zero_division=0)
    print(f"Métricas con umbral={args.umbral:.3f} -> precision={P:.3f} recall={R:.3f} F1={F1:.3f}")

# Guarda resultados
df_out.to_csv(args.out_csv, index=False, encoding="utf-8")
print(f"✅ Guardado: {args.out_csv} | filas={len(df_out)} | umbral={args.umbral:.3f}")

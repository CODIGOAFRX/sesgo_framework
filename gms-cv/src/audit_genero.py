# gms-cv/src/audit_genero.py
import os, argparse, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--pred_csv", default=os.path.join(ROOT, "reports", "predicciones_thr.csv"))
parser.add_argument("--group_col", default="genero")
parser.add_argument("--label_col", default="label")
parser.add_argument("--pred_col", default="preseleccionado")
args = parser.parse_args()

df = pd.read_csv(args.pred_csv)
if not {args.group_col, args.label_col, args.pred_col}.issubset(df.columns):
    raise ValueError("Faltan columnas en el CSV de predicciones.")

def metrics(subdf, y_true_col, y_pred_col):
    y_true = subdf[y_true_col]
    y_pred = subdf[y_pred_col]
    P = precision_score(y_true, y_pred, zero_division=0)
    R = recall_score(y_true, y_pred, zero_division=0)
    F1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fn_rate = fn / (fn + tp) if (fn + tp) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    sel = float(subdf[y_pred_col].mean())   # ← ahora la tasa de selección se calcula sobre el subdataframe correcto
    return dict(precision=P, recall=R, f1=F1, fn_rate=fn_rate, fp_rate=fp_rate, seleccion=sel)

rows = []
for g, part in df.groupby(args.group_col):
    m = metrics(part, args.label_col, args.pred_col)
    rows.append({"grupo": g, **m, "n": len(part)})

# Totales correctamente calculados sobre TODO el dataset
m_tot = metrics(df, args.label_col, args.pred_col)
rows.append({"grupo": "TOTAL", **m_tot, "n": len(df)})

out = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
out_path = os.path.join(ROOT, "reports", "auditoria_genero.csv")
out.to_csv(out_path, index=False, encoding="utf-8")

# Mostrar resumen y deltas clave (M vs H)
print(out)
try:
    h = out[out["grupo"]=="H"].iloc[0]; m = out[out["grupo"]=="M"].iloc[0]
    print(f"\nΔ recall (M - H): {m['recall'] - h['recall']:.3f}")
    print(f"Δ precision (M - H): {m['precision'] - h['precision']:.3f}")
    ratio_seleccion = m["seleccion"] / h["seleccion"] if h["seleccion"] else 0.0
    print(f"Regla del 80% (M/H selección): {ratio_seleccion:.3f}")
except Exception:
    pass
print(f"\n✅ Guardado resumen: {out_path}")

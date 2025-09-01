# gms-cv/src/metrics_genero.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Usamos el framework
from gsmfair.mitigation import EqualizeFprFnr, equalize_rates_from_binary

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRED_PATH = os.path.join(ROOT, "reports", "predicciones_thr.csv")
OUT_TXT   = os.path.join(ROOT, "reports", "metrics_genero.txt")
OUT_CSV   = os.path.join(ROOT, "reports", "predicciones_thr_fair.csv")

# Config knobs
ALPHA = 0.5     # 0.5 equilibra; baja a 0.3 si tu prioridad es FNR gap
GRID  = 401
TREF  = 0.5
SEED  = 7

df = pd.read_csv(PRED_PATH)
required = {"genero", "label", "preseleccionado"}
if not required.issubset(df.columns):
    raise ValueError(f"Faltan columnas en {PRED_PATH}. Requiere: {required}")

def _to_bin(s: pd.Series) -> np.ndarray:
    if s.dtype.kind in "ifu":
        v = pd.to_numeric(s, errors="coerce").fillna(0)
        u = set(np.unique(v.astype(int)))
        if u <= {0,1}: return v.astype(int).to_numpy()
        return (v > float(np.nanmedian(v))).astype(int).to_numpy()
    v = s.astype(str).str.lower().str.strip()
    return v.isin({"1","true","yes","si","sí","y","hire","contratado","pos","positive"}).astype(int).to_numpy()

def _sens_to01(s: pd.Series) -> np.ndarray:
    if s.dtype.kind in "ifu": return _to_bin(s)
    v = s.astype(str).str.upper().str.strip()
    return (v == "M").astype(int).to_numpy()  # M→1, resto→0

def block(sub: pd.DataFrame) -> dict:
    y_true = _to_bin(sub["label"])
    y_pred = _to_bin(sub["preseleccionado"])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    fnr   = fn / (fn + tp) if (fn + tp) else 0.0
    fpr   = fp / (fp + tn) if (fp + tn) else 0.0
    sel   = float(y_pred.mean())
    return dict(n=len(sub), tp=tp, fp=fp, tn=tn, fn=fn,
                precision=prec, recall=rec, f1=f1,
                fn_rate=fnr, fp_rate=fpr, seleccion=sel)

# ===== Baseline (idéntico a tu script) =====
m_tot_base = block(df)
m_h_base   = block(df[df["genero"]=="H"]) if (df["genero"]=="H").any() else None
m_m_base   = block(df[df["genero"]=="M"]) if (df["genero"]=="M").any() else None

# ===== Mitigación =====
y_true = _to_bin(df["label"])
s      = _sens_to01(df["genero"])
y_pred_base = _to_bin(df["preseleccionado"])

# 1) Si existe alguna columna de score continuo, usamos EqualizeFprFnr
score_cols = [c for c in df.columns if c.lower() in
              {"score","prob","proba","prob_pos","prob_1","p1","y_score","predict_proba","decision_score"}]
if score_cols:
    y_scores = pd.to_numeric(df[score_cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pp = EqualizeFprFnr(alpha=ALPHA, grid_size=GRID, clip=(0.01, 0.99))
    # NOTA: no tenemos validación separada; ajustamos y aplicamos en el mismo set (demo)
    pp.fit(y_true_val=y_true, y_scores_val=y_scores, s_val=s, ref_threshold=TREF)
    y_pred_fair = pp.predict(y_scores=y_scores, s=s)
else:
    # 2) No hay scores → usa mitigación binaria por "flips" (reduce gaps sí o sí)
    y_pred_fair, info_flips = equalize_rates_from_binary(
        y_true=y_true, y_pred=y_pred_base, s=s, alpha=ALPHA, seed=SEED
    )
    # Si quieres ver cuántas instancias se cambiaron por grupo:
    print("[gsmfair] flips por grupo:", info_flips["flips"])

# Guardamos CSV con columna mitigada
df_out = df.copy()
df_out["preseleccionado_fair"] = y_pred_fair
df_out.to_csv(OUT_CSV, index=False)

# Recalcula métricas con la columna mitigada
df_post = df.copy()
df_post["preseleccionado"] = y_pred_fair

m_tot_post = block(df_post)
m_h_post   = block(df_post[df_post["genero"]=="H"]) if (df_post["genero"]=="H").any() else None
m_m_post   = block(df_post[df_post["genero"]=="M"]) if (df_post["genero"]=="M").any() else None

# ===== Informe =====
lines = []
lines.append("=== Baseline: métricas globales ===")
for k,v in m_tot_base.items(): lines.append(f"{k}: {v:.6f}" if isinstance(v,float) else f"{k}: {v}")

lines.append("\n=== Fair (postprocesado): métricas globales ===")
for k,v in m_tot_post.items(): lines.append(f"{k}: {v:.6f}" if isinstance(v,float) else f"{k}: {v}")

if m_h_base and m_m_base and m_h_post and m_m_post:
    lines.append("\n=== Baseline: métricas por género ===")
    lines.append("[H]");  [lines.append(f"{k}: {v:.6f}" if isinstance(v,float) else f"{k}: {v}") for k,v in m_h_base.items()]
    lines.append("[M]");  [lines.append(f"{k}: {v:.6f}" if isinstance(v,float) else f"{k}: {v}") for k,v in m_m_base.items()]

    lines.append("\n=== Fair: métricas por género ===")
    lines.append("[H]");  [lines.append(f"{k}: {v:.6f}" if isinstance(v,float) else f"{k}: {v}") for k,v in m_h_post.items()]
    lines.append("[M]");  [lines.append(f"{k}: {v:.6f}" if isinstance(v,float) else f"{k}: {v}") for k,v in m_m_post.items()]

    fpr_gap_base = abs(m_h_base["fp_rate"] - m_m_base["fp_rate"])
    fnr_gap_base = abs(m_h_base["fn_rate"] - m_m_base["fn_rate"])
    fpr_gap_post = abs(m_h_post["fp_rate"] - m_m_post["fp_rate"])
    fnr_gap_post = abs(m_h_post["fn_rate"] - m_m_post["fn_rate"])

    lines.append("\n=== Diferenciales de equidad (gaps) ===")
    lines.append(f"FPR gap BASELINE: {fpr_gap_base:.6f} -> FAIR: {fpr_gap_post:.6f}")
    lines.append(f"FNR gap BASELINE: {fnr_gap_base:.6f} -> FAIR: {fnr_gap_post:.6f}")

    d_rec_base = m_m_base["recall"] - m_h_base["recall"]
    d_pre_base = m_m_base["precision"] - m_h_base["precision"]
    ratio80_base = (m_m_base["seleccion"] / m_h_base["seleccion"]) if m_h_base["seleccion"] else 0.0

    d_rec_post = m_m_post["recall"] - m_h_post["recall"]
    d_pre_post = m_m_post["precision"] - m_h_post["precision"]
    ratio80_post = (m_m_post["seleccion"] / m_h_post["seleccion"]) if m_h_post["seleccion"] else 0.0

    lines.append("\n=== Deltas y Regla del 80% ===")
    lines.append(f"Δ recall (M - H) BASELINE: {d_rec_base:.6f} -> FAIR: {d_rec_post:.6f}")
    lines.append(f"Δ precision (M - H) BASELINE: {d_pre_base:.6f} -> FAIR: {d_pre_post:.6f}")
    lines.append(f"Regla del 80% BASELINE (M/H selección): {ratio80_base:.6f} -> FAIR: {ratio80_post:.6f}")

os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\n✅ Guardado informe: {OUT_TXT}")
print(f"✅ Guardado CSV con columna mitigada: {OUT_CSV}")

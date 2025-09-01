# experiments/01_reweighing_demo.py
from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from gsmfair.metrics import (
    demographic_parity_difference as dpd,
    equal_opportunity_gap as eog,
)
from gsmfair.mitigation import fit_with_reweighing


# ---------- Generación de datos con sesgo controlado ----------
def make_biased_data(n=4000, random_state=7):
    """
    Dataset sintético con disparidad marcada entre grupos sensibles.
    - s ~ Bernoulli(0.5)
    - y se ve afectada por s para inducir distintas tasas positivas/TPR.
    """
    rng = np.random.default_rng(random_state)

    # Problema no trivial (class_sep más bajo)
    X, y = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.6, 0.4],  # prevalencia inicial de clase
        class_sep=0.7,       # ↓ separabilidad para que el modelo no lo tenga tan fácil
        random_state=random_state,
    )

    # Atributo sensible binario
    s = rng.integers(0, 2, size=n)

    # SESGO MÁS FUERTE:
    # - Penalizamos positivos del grupo s=1 (↓ TPR/prevalencia en s=1)
    # - Favorecemos positivos en s=0 (↑ prevalencia en s=0)
    flip_to_0 = (s == 1) & (rng.random(n) < 0.60) & (y == 1)
    flip_to_1 = (s == 0) & (rng.random(n) < 0.02) & (y == 0)
    y = y.copy()
    y[flip_to_0] = 0
    y[flip_to_1] = 1

    return X, y, s


# ---------- Métricas resumidas ----------
def print_report(title, y_true, y_pred, s):
    acc = accuracy_score(y_true, y_pred)
    dpd_val = dpd(y_pred, s)             # paridad demográfica (↓ mejor)
    eog_val = eog(y_true, y_pred, s)     # equal opportunity / TPR gap (↓ mejor)
    print(f"\n=== {title} ===")
    print(f"Accuracy        : {acc:.3f}")
    print(f"DPD (↓ mejor)   : {dpd_val:.3f}")
    print(f"EOG/TPR gap (↓) : {eog_val:.3f}")


def group_rates(y_true, y_pred, s, positive_label=1):
    """Tasas por grupo para entender qué ocurre (tasa de positivos y TPR)."""
    s = np.asarray(s)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for g in np.unique(s):
        m = (s == g)
        pos_rate = (y_pred[m] == positive_label).mean()
        denom = max(1, (y_true[m] == positive_label).sum())
        tpr = ((y_pred[m] == positive_label) & (y_true[m] == positive_label)).sum() / denom
        print(f"  Grupo {g}: positive_rate={pos_rate:.3f} | TPR={tpr:.3f}")


# ---------- Experimento ----------
def main():
    X, y, s = make_biased_data(n=4000, random_state=7)

    # Estratificamos por combinación (y,s) para mantener proporciones en train/test
    strat = np.array([f"{yi}_{si}" for yi, si in zip(y, s)])
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, s, test_size=0.3, random_state=7, stratify=strat
    )

    # ===== Baseline (sin mitigación)
    base = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
    base.fit(X_tr, y_tr)
    y_pred_base = base.predict(X_te)
    print_report("BASELINE (sin mitigación)", y_te, y_pred_base, s_te)
    group_rates(y_te, y_pred_base, s_te)

    # ===== Reweighing (preprocesado con sample_weight)
    # C un poco mayor → menos regularización → los sample_weight influyen más
    rw_model = LogisticRegression(max_iter=3000, C=2.0, solver="lbfgs")
    est_rw, info = fit_with_reweighing(rw_model, X_tr, y_tr, s_tr, normalize="mean")
    y_pred_rw = est_rw.predict(X_te)
    print_report("REWEIGHING (con sample_weight)", y_te, y_pred_rw, s_te)
    group_rates(y_te, y_pred_rw, s_te)

    # Información adicional
    w = info["sample_weight"]
    print(f"\n[Info] Peso medio (train): {np.mean(w):.3f}  |  min={np.min(w):.3f}  max={np.max(w):.3f}")
    print("[Info] Mapa de pesos (g|y):", info["reweighing_details"]["w_map"])


if __name__ == "__main__":
    main()

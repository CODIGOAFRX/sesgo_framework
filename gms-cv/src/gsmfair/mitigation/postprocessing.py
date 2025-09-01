# gms-cv/src/gsmfair/mitigation/postprocessing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple
import numpy as np

def _fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Devuelve (FPR, FNR). Maneja divisiones por cero."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    neg = (y_true != 1)
    pos = (y_true == 1)

    fp = np.sum((y_pred == 1) & neg)
    tn = np.sum((y_pred != 1) & neg)
    fn = np.sum((y_pred != 1) & pos)
    tp = np.sum((y_pred == 1) & pos)

    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))
    return float(fpr), float(fnr)

def _to_1d_scores(a: Iterable[float], n_target: int | None = None) -> np.ndarray:
    """
    Convierte "a" en vector 1D de longitud n_samples.
    Casos soportados:
      - (n,) ya 1D
      - (n,2) probas -> usa la columna de la clase positiva
      - (2,n) probas transpuestas -> usa la segunda fila
      - (n,k) cualquier otro -> aplana en 1D
    """
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim >= 2:
        # Probabilidades típicas (n,2)
        if arr.shape[-1] == 2 and arr.shape[0] != 2:
            return arr[:, 1].astype(float)
        # Probabilidades transpuestas (2,n)
        if arr.shape[0] == 2 and (n_target is None or arr.shape[1] == n_target):
            return arr[1, :].astype(float)
        # Si es (n,1) o similar, aplana
        return arr.reshape(-1).astype(float)
    # fallback
    return np.asarray(arr, dtype=float).reshape(-1)

def _pred_from_scores(scores: np.ndarray, thr: float) -> np.ndarray:
    scores = np.asarray(scores).ravel()
    return (scores >= thr).astype(int)

@dataclass
class EqualizeFprFnr:
    """
    Postprocesado: fija un UMBRAL por grupo sensible para acercar FPR/FNR entre grupos.
    Minimiza L = alpha*|FPR_g - FPR_ref| + (1-alpha)*|FNR_g - FNR_ref| por grupo g.
    """
    alpha: float = 0.5          # 1.0 -> prioriza FPR; 0.0 -> prioriza FNR
    grid_size: int = 201        # resolución de búsqueda de umbrales
    clip: Tuple[float, float] = (0.01, 0.99)  # evita extremos

    thresholds_: Dict[Any, float] | None = None
    ref_: Dict[str, float] | None = None

    def fit(
        self,
        y_true_val: Iterable[int],
        y_scores_val: Iterable[float],
        s_val: Iterable[Any],
        *,
        ref_threshold: float = 0.5
    ) -> "EqualizeFprFnr":
        y_true_val = np.asarray(y_true_val).ravel()
        s_val = np.asarray(s_val).ravel()
        # Fuerza scores a 1D consistente con y_true_val
        y_scores_val = _to_1d_scores(y_scores_val, n_target=len(y_true_val))

        if not (len(y_true_val) == len(y_scores_val) == len(s_val)):
            raise ValueError(
                f"Longitudes distintas: y_true={len(y_true_val)}, "
                f"scores={len(y_scores_val)}, s={len(s_val)}"
            )

        # Referencia con umbral global
        y_pred_ref = _pred_from_scores(y_scores_val, ref_threshold)

        groups = np.unique(s_val)
        fprs, fnrs = [], []
        for g in groups:
            m = (s_val == g)
            fpr_g, fnr_g = _fpr_fnr(y_true_val[m], y_pred_ref[m])
            fprs.append(fpr_g); fnrs.append(fnr_g)
        fpr_ref = float(np.mean(fprs)); fnr_ref = float(np.mean(fnrs))
        self.ref_ = {"FPR": fpr_ref, "FNR": fnr_ref}

        # Rejilla de búsqueda
        lo, hi = self.clip
        grid = np.linspace(lo, hi, int(self.grid_size))

        thresholds: Dict[Any, float] = {}
        for g in groups:
            m = (s_val == g)
            ys = y_scores_val[m]
            yt = y_true_val[m]
            if ys.size == 0:
                thresholds[g] = ref_threshold
                continue

            best_thr = ref_threshold
            best_loss = float("inf")
            for t in grid:
                yp = _pred_from_scores(ys, t)
                fpr_g, fnr_g = _fpr_fnr(yt, yp)
                loss = self.alpha * abs(fpr_g - fpr_ref) + (1 - self.alpha) * abs(fnr_g - fnr_ref)
                if loss < best_loss:
                    best_loss = loss
                    best_thr = float(t)
            thresholds[g] = best_thr

        self.thresholds_ = thresholds
        return self

    def predict(self, y_scores: Iterable[float], s: Iterable[Any]) -> np.ndarray:
        assert self.thresholds_ is not None, "Llama a fit() primero."
        s = np.asarray(s).ravel()
        y_scores = _to_1d_scores(y_scores, n_target=len(s))

        if len(y_scores) != len(s):
            raise ValueError(f"Longitudes distintas en predict: scores={len(y_scores)}, s={len(s)}")

        y_pred = np.zeros_like(y_scores, dtype=int)
        for g, thr in self.thresholds_.items():
            m = (s == g)
            if np.any(m):
                y_pred[m] = _pred_from_scores(y_scores[m], thr)
        return y_pred

    def get_thresholds(self) -> Dict[Any, float]:
        return dict(self.thresholds_) if self.thresholds_ else {}

    def get_reference(self) -> Dict[str, float]:
        return dict(self.ref_) if self.ref_ else {}
    
# --- NUEVO: mitigación cuando solo hay predicción binaria (sin scores) ---
from typing import Tuple, Any, Dict

def equalize_rates_from_binary(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    s: Iterable[Any],
    *,
    alpha: float = 0.5,   # 1→prioriza FPR, 0→prioriza FNR
    seed: int = 7
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ajusta y_pred 'flipping' una fracción mínima de FPs/FNs por grupo para
    acercar FPR/FNR al mejor grupo (no se empeora a nadie).
    Devuelve (y_pred_mitigado, info).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    s = np.asarray(s).ravel()

    assert len(y_true) == len(y_pred) == len(s), "Longitudes incompatibles"

    def fpr_fnr(yt, yp):
        neg = (yt == 0); pos = (yt == 1)
        fp = np.sum((yp == 1) & neg); tn = np.sum((yp == 0) & neg)
        fn = np.sum((yp == 0) & pos); tp = np.sum((yp == 1) & pos)
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        return fpr, fnr, int(fp), int(tn), int(fn), int(tp)

    # Métricas 'antes' por grupo
    groups = np.unique(s)
    before, fprs, fnrs = {}, [], []
    for g in groups:
        m = (s == g)
        fpr, fnr, fp, tn, fn, tp = fpr_fnr(y_true[m], y_pred[m])
        before[str(g)] = {"FPR": fpr, "FNR": fnr, "FP": fp, "TN": tn, "FN": fn, "TP": tp, "n": int(m.sum())}
        fprs.append(fpr); fnrs.append(fnr)

    # Referencia: el mejor grupo en cada tasa (no empeoramos a nadie)
    fpr_ref = float(np.min(fprs))
    fnr_ref = float(np.min(fnrs))

    y_new = y_pred.copy()
    flips: Dict[str, Dict[str, int]] = {}

    # Ajuste por grupo
    for g in groups:
        m = (s == g)
        idx_g = np.where(m)[0]
        yt, yp = y_true[idx_g], y_new[idx_g]

        fpr, fnr, fp, tn, fn, tp = fpr_fnr(yt, yp)

        # Reducir FPR (convertir FP -> 0)
        need_fpr = int(np.ceil(alpha * max(0.0, fpr - fpr_ref) * (fp + tn)))
        if need_fpr > 0:
            cand_fp = idx_g[(y_true[idx_g] == 0) & (y_new[idx_g] == 1)]  # FPs
            k = int(min(need_fpr, cand_fp.size))
            if k > 0:
                sel = rng.choice(cand_fp, size=k, replace=False)
                y_new[sel] = 0
        else:
            k = 0

        # Reducir FNR (convertir FN -> 1)
        need_fnr = int(np.ceil((1 - alpha) * max(0.0, fnr - fnr_ref) * (fn + tp)))
        if need_fnr > 0:
            cand_fn = idx_g[(y_true[idx_g] == 1) & (y_new[idx_g] == 0)]  # FNs
            k2 = int(min(need_fnr, cand_fn.size))
            if k2 > 0:
                sel2 = rng.choice(cand_fn, size=k2, replace=False)
                y_new[sel2] = 1
        else:
            k2 = 0

        flips[str(g)] = {"flip_FP_to_0": int(k), "flip_FN_to_1": int(k2)}

    # Métricas 'después'
    after = {}
    for g in groups:
        m = (s == g)
        fpr, fnr, fp, tn, fn, tp = fpr_fnr(y_true[m], y_new[m])
        after[str(g)] = {"FPR": fpr, "FNR": fnr, "FP": fp, "TN": tn, "FN": fn, "TP": tp, "n": int(m.sum())}

    info = {"ref": {"FPR": fpr_ref, "FNR": fnr_ref}, "before": before, "after": after, "flips": flips}
    return y_new, info


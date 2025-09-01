# src/gsmfair/mitigation/shortcuts.py
from __future__ import annotations
import numpy as np
from typing import Any, Iterable
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from .postprocessing import EqualizeFprFnr

def _get_scores(clf: BaseEstimator, X) -> np.ndarray:
    """Devuelve scores en [0,1] para clase positiva."""
    # 1) predict_proba si existe
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    # 2) decision_function → escalar a [0,1]
    if hasattr(clf, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler
        s = clf.decision_function(X).reshape(-1, 1)
        return MinMaxScaler().fit_transform(s).ravel()
    # 3) fallback: calibrar el propio modelo
    base = clf
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    # OJO: para calibrar necesitaríamos re-entrenar; este atajo asume que
    # si no hay proba/decision_function ya vendrás con el modelo "cal".
    raise AttributeError("El clasificador no expone scores. Usa CalibratedClassifierCV al entrenar.")

def equalize_fpr_fnr_predict(
    clf: BaseEstimator,
    X_val,
    y_val: Iterable[int],
    s_val: Iterable[Any],
    X_test,
    s_test: Iterable[Any],
    *,
    alpha: float = 0.5,
    ref_threshold: float = 0.5,
    grid_size: int = 201
) -> np.ndarray:
    """
    Devuelve y_pred_post en TEST aplicando umbral por grupo para acercar FPR/FNR.
    """
    y_scores_val = _get_scores(clf, X_val)
    y_scores_test = _get_scores(clf, X_test)

    pp = EqualizeFprFnr(alpha=alpha, grid_size=grid_size, clip=(0.01, 0.99))
    pp.fit(y_true_val=y_val, y_scores_val=y_scores_val, s_val=np.asarray(list(s_val)), ref_threshold=ref_threshold)
    y_pred_post = pp.predict(y_scores=y_scores_test, s=np.asarray(list(s_test)))
    return y_pred_post

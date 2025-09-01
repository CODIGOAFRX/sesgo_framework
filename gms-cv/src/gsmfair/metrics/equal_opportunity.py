# src/gsmfair/metrics/equal_opportunity.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple
import numpy as np

def equal_opportunity_gap(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    s: Iterable[Any],
    positive_label: Any = 1,
    return_details: bool = False
) -> float | Tuple[float, Dict[Any, Dict[str, float]]]:
    """
    Calcula la brecha de Oportunidad Igualitaria (TPR gap):
        gap = max_g TPR_g - min_g TPR_g
    donde TPR_g = TP_g / (TP_g + FN_g), condicionado a s = g y y_true = positive_label.

    Parámetros
    ----------
    y_true : iterable
        Etiquetas reales (binarias; se usa `positive_label` como positiva).
    y_pred : iterable
        Predicciones (binarias o probabilísticas ya umbralizadas).
    s : iterable
        Atributo sensible por individuo (p. ej., 'M'/'F', 0/1, etc.).
    positive_label : Any, opcional (por defecto=1)
        Etiqueta considerada positiva.
    return_details : bool, opcional
        Si True, devuelve (gap, detalles_por_grupo).

    Devuelve
    --------
    float o (float, dict)

    Errores
    -------
    ValueError si longitudes no coinciden o no hay, al menos, dos grupos con positivos reales.

    Notas
    -----
    - Un gap cercano a 0 implica igualdad de oportunidad (mismo TPR entre grupos).
    """
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    s = np.asarray(list(s))

    if not (len(y_true) == len(y_pred) == len(s)):
        raise ValueError("`y_true`, `y_pred` y `s` deben tener la misma longitud.")

    groups = np.unique(s)
    tprs = []
    details: Dict[Any, Dict[str, float]] = {}

    for g in groups:
        mask_g = (s == g)
        mask_pos = (y_true == positive_label) & mask_g
        pos_g = mask_pos.sum()
        if pos_g == 0:
            # Sin positivos reales en el grupo -> no define TPR
            continue
        tp_g = np.sum((y_pred == positive_label) & mask_pos)
        tpr_g = float(tp_g) / float(pos_g)
        tprs.append(tpr_g)
        details[g] = {"positives": float(pos_g), "tp": float(tp_g), "tpr": tpr_g}

    if len(tprs) < 2:
        raise ValueError("Se requieren al menos dos grupos con positivos reales para calcular el TPR gap.")

    gap = float(np.max(tprs) - np.min(tprs))
    return (gap, details) if return_details else gap

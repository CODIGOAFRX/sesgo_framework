# src/gsmfair/metrics/demographic_parity.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple
import numpy as np

def demographic_parity_difference(
    y_pred: Iterable[Any],
    s: Iterable[Any],
    positive_label: Any = 1,
    return_details: bool = False
) -> float | Tuple[float, Dict[Any, Dict[str, float]]]:
    """
    Calcula la Demographic Parity Difference (DPD):
        DPD = max_g P(ŷ = pos | s = g) - min_g P(ŷ = pos | s = g)

    Parámetros
    ----------
    y_pred : iterable
        Predicciones binaria(s). Se considerará `positive_label` como la clase positiva.
    s : iterable
        Atributo sensible por individuo (p. ej., 'M'/'F', 0/1, etc.). Puede haber 2+ grupos.
    positive_label : Any, opcional (por defecto=1)
        Etiqueta que se considera positiva.
    return_details : bool, opcional
        Si True, devuelve (dpd, detalles_por_grupo).

    Devuelve
    --------
    float o (float, dict)
        La DPD en [0,1]. Si `return_details=True`, también un dict con tasas y tamaños por grupo.

    Errores
    -------
    ValueError si longitudes no coinciden o no hay grupos válidos.

    Notas
    -----
    - DPD=0 implica misma tasa de positivos predichos en todos los grupos.
    - Valores cercanos a 0 indican mayor paridad demográfica (no garantiza ausencia de otros sesgos).
    """
    y_pred = np.asarray(list(y_pred))
    s = np.asarray(list(s))
    if y_pred.shape[0] != s.shape[0]:
        raise ValueError("`y_pred` y `s` deben tener la misma longitud.")

    # Asegurar binariedad respecto a positive_label
    # (No forzamos tipos: solo comparamos igualdad con positive_label)
    groups, counts = np.unique(s, return_counts=True)
    if groups.size < 2:
        raise ValueError("Se requieren al menos dos grupos en `s`.")

    rates = []
    details: Dict[Any, Dict[str, float]] = {}
    for g, n in zip(groups, counts):
        mask = (s == g)
        if not np.any(mask):
            continue
        num_pos = np.sum(y_pred[mask] == positive_label)
        rate = float(num_pos) / float(mask.sum())
        rates.append(rate)
        details[g] = {"n": float(mask.sum()), "pos": float(num_pos), "positive_rate": rate}

    if len(rates) < 2:
        raise ValueError("No hay suficientes grupos con observaciones para calcular DPD.")

    dpd = float(np.max(rates) - np.min(rates))
    return (dpd, details) if return_details else dpd

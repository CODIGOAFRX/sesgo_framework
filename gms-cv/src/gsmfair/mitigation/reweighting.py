# src/gsmfair/mitigation/reweighting.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple
import numpy as np

def reweighing_weights(
    y_true: Iterable[Any],
    s: Iterable[Any],
    normalize: str | bool = "mean"
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Calcula pesos por observación según la técnica de reponderación:
        w(g,y) = P(y) * P(g) / P(g,y)

    Parámetros
    ----------
    y_true : iterable
        Etiquetas reales (binarias o categóricas).
    s : iterable
        Atributo sensible por individuo (p. ej., 'M'/'F', 0/1, etc.).
    normalize : {'mean','sum', False}, opcional
        - 'mean': escala para que la media de los pesos sea 1 (por defecto).
        - 'sum' : escala para que la suma de pesos sea N.
        - False  : no normaliza.

    Devuelve
    --------
    weights : np.ndarray de shape (n_samples,)
        Peso por observación, en el mismo orden de `y_true`/`s`.
    details : dict
        Probabilidades y pesos por (grupo, etiqueta) para trazabilidad.

    Notas
    -----
    - Si P(g,y)=0 para alguna combinación, su peso se fija a 0.
    - Normalizar ayuda a que los estimadores de scikit-learn interpreten
      los pesos en una escala razonable.
    """
    y = np.asarray(list(y_true))
    sens = np.asarray(list(s))
    if y.shape[0] != sens.shape[0]:
        raise ValueError("`y_true` y `s` deben tener la misma longitud.")

    n = y.shape[0]
    labels = np.unique(y)
    groups = np.unique(sens)

    # Probabilidades marginales y conjuntas
    P_y = {lbl: np.mean(y == lbl) for lbl in labels}
    P_g = {g: np.mean(sens == g) for g in groups}
    P_gy = {(g, lbl): np.mean((sens == g) & (y == lbl)) for g in groups for lbl in labels}

    # Mapa de pesos por combinación (g, y)
    w_map: Dict[Tuple[Any, Any], float] = {}
    for g in groups:
        for lbl in labels:
            pgy = P_gy[(g, lbl)]
            w_map[(g, lbl)] = 0.0 if pgy == 0.0 else (P_y[lbl] * P_g[g]) / pgy

    # Pesos por observación
    weights = np.array([w_map[(g, lbl)] for g, lbl in zip(sens, y)], dtype=float)

    # Normalización
    if normalize:
        if normalize == "mean":
            denom = np.mean(weights[weights > 0]) if np.any(weights > 0) else 1.0
            weights = weights / denom
        elif normalize == "sum":
            ssum = weights.sum()
            weights = (weights * n / ssum) if ssum > 0 else weights
        else:
            raise ValueError("`normalize` debe ser 'mean', 'sum' o False.")

    # Detalles para auditoría
    details: Dict[str, Dict[str, float]] = {
        "P_y": {str(k): float(v) for k, v in P_y.items()},
        "P_g": {str(k): float(v) for k, v in P_g.items()},
        "P_g_y": {f"{g}|{lbl}": float(v) for (g, lbl), v in P_gy.items()},
        "w_map": {f"{g}|{lbl}": float(w) for (g, lbl), w in w_map.items()},
    }
    return weights, details

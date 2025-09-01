# src/gsmfair/mitigation/helpers.py
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from sklearn.base import BaseEstimator, clone

from .reweighting import reweighing_weights

def fit_with_reweighing(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    *,
    normalize: str | bool = "mean",
    **fit_kwargs
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Entrena un estimador de scikit-learn aplicando reponderación (preprocesado).
    Calcula sample_weight = w(g,y) y llama a est.fit(X, y, sample_weight=...).

    Parámetros
    ----------
    estimator : BaseEstimator
        Modelo o Pipeline de scikit-learn (se clona para no mutar el original).
    X : array (n_samples, n_features)
        Matriz de características.
    y : array (n_samples,)
        Etiquetas reales (binarias o categóricas).
    s : array (n_samples,)
        Atributo sensible por individuo (p.ej., 0/1 o 'M'/'F').
    normalize : {'mean','sum', False}
        Cómo normalizar los pesos (por defecto 'mean').
    **fit_kwargs :
        Parámetros extra que se pasan a `estimator.fit`.

    Devuelve
    --------
    est_fit : BaseEstimator
        Estimador ya entrenado con reponderación.
    info : dict
        Diccionario con 'sample_weight' y detalles del cálculo de pesos.
    """
    est = clone(estimator)
    w, details = reweighing_weights(y_true=y, s=s, normalize=normalize)
    est.fit(X, y, sample_weight=w, **fit_kwargs)
    info = {"sample_weight": w, "reweighing_details": details}
    return est, info

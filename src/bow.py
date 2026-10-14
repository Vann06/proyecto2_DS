"""Utilidades reproducibles para representar los resúmenes con Bag of Words."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.evaluation import regression_metrics  # noqa: F401  (se re-exporta para 06 y 07)
from src.tokenization import tokenize_text


TARGETS = ("content", "wording")


def build_bow_vectorizer(
    *,
    min_df: int = 1,
    max_df: float = 0.98,
    binary: bool = False,
) -> CountVectorizer:
    """Crea un vectorizador BoW compatible con la tokenización del proyecto.

    El vocabulario siempre debe ajustarse únicamente con los datos de
    entrenamiento cuando la representación se usa para modelar.
    """
    if not isinstance(min_df, int) or min_df < 1:
        raise ValueError("min_df debe ser un entero mayor o igual que 1.")
    if not 0 < max_df <= 1:
        raise ValueError("max_df debe pertenecer al intervalo (0, 1].")

    return CountVectorizer(
        tokenizer=tokenize_text,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
        dtype=np.float64,
    )


def vocabulary_diagnostic(
    texts: Sequence[str] | pd.Series,
    min_document_frequencies: Iterable[int] = (1, 2, 5, 10),
    *,
    max_df: float = 0.98,
) -> pd.DataFrame:
    """Resume tamaño, densidad y dispersión de varias matrices BoW."""
    documents = pd.Series(texts, dtype="object")
    if documents.empty:
        raise ValueError("Se necesita al menos un documento.")
    if documents.isna().any() or not documents.map(lambda value: isinstance(value, str)).all():
        raise ValueError("Todos los documentos deben ser cadenas sin valores faltantes.")

    rows = []
    for min_df in min_document_frequencies:
        vectorizer = build_bow_vectorizer(min_df=int(min_df), max_df=max_df)
        matrix = vectorizer.fit_transform(documents)
        total_cells = matrix.shape[0] * matrix.shape[1]
        density = matrix.nnz / total_cells if total_cells else 0.0
        rows.append(
            {
                "min_df": int(min_df),
                "documentos": matrix.shape[0],
                "vocabulario": matrix.shape[1],
                "valores_no_cero": matrix.nnz,
                "densidad": density,
                "dispersion": 1.0 - density,
                "variables_activas_por_documento": matrix.nnz / matrix.shape[0],
            }
        )
    return pd.DataFrame(rows)


def coefficient_table(
    feature_names: Sequence[str],
    coefficients: np.ndarray,
    targets: Sequence[str] = TARGETS,
    *,
    top_n: int = 12,
) -> pd.DataFrame:
    """Devuelve los coeficientes positivos y negativos más grandes por objetivo."""
    names = np.asarray(feature_names)
    weights = np.asarray(coefficients, dtype=float)
    if weights.shape != (len(targets), len(names)):
        raise ValueError("Las dimensiones de coeficientes, objetivos y vocabulario no coinciden.")
    if top_n < 1:
        raise ValueError("top_n debe ser mayor o igual que 1.")

    rows = []
    for target_index, target in enumerate(targets):
        order = np.argsort(weights[target_index])
        selected = list(order[:top_n]) + list(order[-top_n:])
        for feature_index in selected:
            value = weights[target_index, feature_index]
            rows.append(
                {
                    "objetivo": target,
                    "token": names[feature_index],
                    "coeficiente": value,
                    "dirección": "positiva" if value >= 0 else "negativa",
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["objetivo", "coeficiente"], ascending=[True, True])
        .reset_index(drop=True)
    )

"""Representación TF-IDF y similitud léxica para resúmenes de CommonLit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from src.tokenization import tokenize_text


def build_tfidf_vectorizer(
    *, min_df: int = 1, max_df: float = 0.98,
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int | None = None, sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Crea un vectorizador con la tokenización compartida del proyecto."""
    if not isinstance(min_df, int) or min_df < 1:
        raise ValueError("min_df debe ser un entero mayor o igual que 1.")
    if not 0 < max_df <= 1:
        raise ValueError("max_df debe pertenecer al intervalo (0, 1].")
    if not 1 <= ngram_range[0] <= ngram_range[1]:
        raise ValueError("ngram_range debe indicar un intervalo válido.")
    return TfidfVectorizer(
        tokenizer=tokenize_text, preprocessor=None, token_pattern=None,
        lowercase=False, min_df=min_df, max_df=max_df,
        ngram_range=ngram_range, max_features=max_features,
        sublinear_tf=sublinear_tf, smooth_idf=True, norm="l2",
        dtype=np.float64,
    )


def paired_cosine_similarity(summary_matrix, prompt_matrix) -> np.ndarray:
    """Calcula el coseno por pares sin crear una matriz de todas las parejas."""
    if summary_matrix.shape != prompt_matrix.shape:
        raise ValueError("Los resúmenes y sus fuentes deben tener la misma forma.")
    dot = np.asarray(summary_matrix.multiply(prompt_matrix).sum(axis=1)).ravel()
    summary_norm = np.sqrt(np.asarray(summary_matrix.power(2).sum(axis=1)).ravel())
    prompt_norm = np.sqrt(np.asarray(prompt_matrix.power(2).sum(axis=1)).ravel())
    denominator = summary_norm * prompt_norm
    similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    return np.clip(similarity, 0.0, 1.0)


def transform_prompt_texts(vectorizer, texts: pd.Series):
    """Transforma cada fuente distinta una vez y conserva el orden de las filas."""
    positions, unique_texts = pd.factorize(texts, sort=False)
    if (positions < 0).any():
        raise ValueError("Los textos fuente no deben contener valores faltantes.")
    return vectorizer.transform(unique_texts)[positions]


def validate_text_pairs(frame: pd.DataFrame) -> None:
    """Valida los textos requeridos para entrenamiento y predicción."""
    if not isinstance(frame, pd.DataFrame) or not {"text", "prompt_text"}.issubset(frame.columns):
        raise ValueError("Se requieren las columnas text y prompt_text.")
    if frame.empty:
        raise ValueError("Se requiere al menos un resumen.")
    for column in ["text", "prompt_text"]:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"La columna {column} debe contener textos no vacíos.")


class SummaryTfidfFeatures(TransformerMixin, BaseEstimator):
    """Ajusta TF-IDF con los resúmenes de entrenamiento y transforma sus fuentes.

    La fuente de cada entrada se transforma con el vocabulario y el IDF ya
    ajustados. Los textos de validación no se incorporan al corpus de ajuste.
    """

    def __init__(self, min_df=1, max_df=0.98, ngram_range=(1, 1),
                 sublinear_tf=True, include_similarity=False):
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.include_similarity = include_similarity

    def fit(self, X, y=None):
        validate_text_pairs(X)
        self.vectorizer_ = build_tfidf_vectorizer(
            min_df=self.min_df, max_df=self.max_df,
            ngram_range=self.ngram_range, sublinear_tf=self.sublinear_tf,
        )
        self.vectorizer_.fit(X["text"])
        return self

    def transform(self, X):
        check_is_fitted(self, "vectorizer_")
        validate_text_pairs(X)
        summaries = self.vectorizer_.transform(X["text"])
        if not self.include_similarity:
            return summaries
        prompts = transform_prompt_texts(self.vectorizer_, X["prompt_text"])
        similarity = paired_cosine_similarity(summaries, prompts)
        return sparse.hstack([summaries, sparse.csr_matrix(similarity[:, None])], format="csr")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "vectorizer_")
        names = self.vectorizer_.get_feature_names_out()
        if self.include_similarity:
            names = np.append(names, "tfidf_prompt_similarity")
        return names


class SummaryBowFeatures(TransformerMixin, BaseEstimator):
    """Adapta BoW a las mismas entradas y particiones que TF-IDF."""

    def __init__(self, min_df=1, max_df=0.98, binary=True):
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary

    def fit(self, X, y=None):
        from src.bow import build_bow_vectorizer

        validate_text_pairs(X)
        self.vectorizer_ = build_bow_vectorizer(
            min_df=self.min_df, max_df=self.max_df, binary=self.binary,
        )
        self.vectorizer_.fit(X["text"])
        return self

    def transform(self, X):
        check_is_fitted(self, "vectorizer_")
        validate_text_pairs(X)
        return self.vectorizer_.transform(X["text"])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "vectorizer_")
        return self.vectorizer_.get_feature_names_out()


def predict_scores(model, text: str, prompt_text: str) -> dict[str, float]:
    """Predice las dos puntuaciones con el preprocesamiento del modelo guardado."""
    frame = pd.DataFrame({"text": [text], "prompt_text": [prompt_text]})
    validate_text_pairs(frame)
    prediction = np.asarray(model.predict(frame), dtype=float)
    if prediction.shape != (1, 2) or not np.isfinite(prediction).all():
        raise ValueError("El modelo debe producir dos puntuaciones finitas.")
    return {"content": float(prediction[0, 0]), "wording": float(prediction[0, 1])}

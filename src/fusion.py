"""Fusión mínima: features manuales + similitud con el texto fuente + un LightGBM por objetivo.

Es el modelo elegido para la aplicación en el notebook 12. Las mismas funciones construyen las features al entrenar
y al predecir, así que un resumen nuevo pasa exactamente por el mismo preprocesamiento que los de entrenamiento.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from functools import lru_cache

import numpy as np
import pandas as pd

from src.features import manual_features
from src.ngrams import generate_ngrams
from src.tokenization import TOKEN_PATTERN, tokenize_text

TARGETS: tuple[str, str] = ("content", "wording")
SPACY_MODEL: str = "en_core_web_md"
# Solo se necesitan los vectores de palabras; el resto del pipeline de spaCy se omite, igual que en el notebook 08.
SPACY_EXCLUDE: list[str] = ["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "senter"]
FEATURE_COLUMNS: list[str] = [
    "token_count", "log_token_count", "unique_token_count", "lexical_diversity",
    "palabras_prompt_overlap", "bigramas_prompt_overlap", "trigramas_prompt_overlap", "cos_sim",
]
# Hiperparámetros elegidos con Optuna en el notebook 11 (búsqueda final con los 4 prompts)
FINAL_PARAMS: dict[str, dict] = {
    "content": dict(n_estimators=500, learning_rate=0.018016236022416487, num_leaves=7, min_child_samples=22,
                    subsample=0.7540091948632973, subsample_freq=1, colsample_bytree=0.8975749734973388,
                    reg_lambda=5.24101873994983, reg_alpha=0.029467455856868555),
    "wording": dict(n_estimators=300, learning_rate=0.01434854980801475, num_leaves=11, min_child_samples=18,
                    subsample=0.6042393742749281, subsample_freq=1, colsample_bytree=0.49384981874557793,
                    reg_lambda=0.9849259076056183, reg_alpha=0.012944825580779884),
}
FIXED_PARAMS: dict = dict(random_state=42, verbose=-1, n_jobs=-1)


@lru_cache(maxsize=1)
def load_nlp():
    """Carga el modelo de spaCy una sola vez por proceso."""
    import spacy

    try:
        return spacy.load(SPACY_MODEL, exclude=SPACY_EXCLUDE)
    except OSError as error:
        raise OSError(f"Falta el modelo de spaCy. Instálelo con: python -m spacy download {SPACY_MODEL}") from error


def document_vectors(texts: list[str], nlp: Callable | None = None) -> np.ndarray:
    """Vector de cada texto: el promedio de los vectores de sus palabras en spaCy."""
    nlp = nlp or load_nlp()
    return np.vstack([doc.vector for doc in nlp.pipe(texts, batch_size=256)])


def validate_inputs(summaries: pd.DataFrame, prompts: pd.DataFrame) -> None:
    """Comprueba columnas, textos vacíos y que cada resumen tenga su texto fuente."""
    for frame, columns, name in [(summaries, ["prompt_id", "text"], "summaries"),
                                 (prompts, ["prompt_id", "prompt_text"], "prompts")]:
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f"A {name} le faltan las columnas {sorted(missing)}.")
    texts = pd.concat([summaries["text"], prompts["prompt_text"]])
    if texts.isna().any() or not all(isinstance(text, str) and tokenize_text(text) for text in texts):
        raise ValueError("Los resúmenes y los textos fuente deben tener al menos una palabra.")
    if prompts["prompt_id"].duplicated().any():
        raise ValueError("Cada prompt_id debe aparecer una sola vez en prompts.")
    unknown = set(summaries["prompt_id"]).difference(prompts["prompt_id"])
    if unknown:
        raise ValueError(f"Hay resúmenes sin texto fuente: {sorted(unknown)}.")


def build_features(summaries: pd.DataFrame, prompts: pd.DataFrame, nlp: Callable | None = None) -> pd.DataFrame:
    """Matriz de la fusión mínima: una fila por resumen, en el orden de ``FEATURE_COLUMNS``.

    ``summaries`` necesita ``prompt_id`` y ``text``; ``prompts``, ``prompt_id`` y ``prompt_text``.
    """
    validate_inputs(summaries, prompts)
    features = manual_features(summaries, prompts)
    emb_summaries = document_vectors(summaries["text"].tolist(), nlp)
    emb_prompts = document_vectors(prompts["prompt_text"].tolist(), nlp)
    prompt_position = {prompt_id: i for i, prompt_id in enumerate(prompts["prompt_id"])}
    emb_prompt_per_row = emb_prompts[[prompt_position[prompt_id] for prompt_id in summaries["prompt_id"]]]
    features["cos_sim"] = (emb_summaries * emb_prompt_per_row).sum(axis=1) / (
        np.linalg.norm(emb_summaries, axis=1) * np.linalg.norm(emb_prompt_per_row, axis=1) + 1e-9)
    return features[FEATURE_COLUMNS]


def train_fusion(features: pd.DataFrame, y: np.ndarray, params: dict[str, dict] | None = None) -> dict:
    """Entrena un LightGBM por objetivo y devuelve el paquete que se guarda con joblib."""
    import lightgbm
    import sklearn

    from lightgbm import LGBMRegressor

    params = params or FINAL_PARAMS
    X = features[FEATURE_COLUMNS]
    models = {target: LGBMRegressor(**params[target], **FIXED_PARAMS).fit(X, y[:, j])
              for j, target in enumerate(TARGETS)}
    return {
        "models": models,
        "feature_columns": FEATURE_COLUMNS,
        "params": params,
        "spacy_model": SPACY_MODEL,
        "versions": {"lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__},
        "training": {"summaries": len(X),
                     "token_count_range": [int(X["token_count"].min()), int(X["token_count"].max())]},
    }


def predict_frame(bundle: dict, summaries: pd.DataFrame, prompts: pd.DataFrame,
                  nlp: Callable | None = None) -> pd.DataFrame:
    """Predice content y wording para varios resúmenes; devuelve una fila por resumen, en el mismo orden."""
    features = build_features(summaries, prompts, nlp)[bundle["feature_columns"]]
    result = pd.DataFrame(index=summaries.index)
    for target in TARGETS:
        result[target] = bundle["models"][target].predict(features)
    low, high = bundle["training"]["token_count_range"]
    result["advertencia"] = [length_warning(count, low, high) for count in features["token_count"]]
    return result


def predict_scores(bundle: dict, text: str, prompt_text: str, nlp: Callable | None = None) -> dict:
    """Predice content y wording de un solo resumen a partir de su texto fuente."""
    result = predict_frame(bundle, pd.DataFrame({"prompt_id": ["nuevo"], "text": [text]}),
                           pd.DataFrame({"prompt_id": ["nuevo"], "prompt_text": [prompt_text]}), nlp)
    row = result.iloc[0]
    return {"content": float(row["content"]), "wording": float(row["wording"]), "advertencia": row["advertencia"]}


def length_warning(token_count: int, low: int, high: int) -> str | None:
    """Aviso cuando el resumen queda fuera del rango de largo de los resúmenes de entrenamiento."""
    if low <= token_count <= high:
        return None
    return (f"El resumen tiene {token_count} palabras y el modelo se entrenó con resúmenes de {low} a {high}: "
            "la predicción es menos confiable.")


def explain_scores(bundle: dict, text: str, prompt_text: str, nlp: Callable | None = None) -> dict:
    """Predicción de un resumen con sus features y cuánto aportó cada una a cada nota (valores SHAP de LightGBM).

    En cada objetivo, ``base`` más la suma de ``contribuciones`` es exactamente la nota predicha.
    """
    summaries = pd.DataFrame({"prompt_id": ["nuevo"], "text": [text]})
    prompts = pd.DataFrame({"prompt_id": ["nuevo"], "prompt_text": [prompt_text]})
    features = build_features(summaries, prompts, nlp)[bundle["feature_columns"]]
    low, high = bundle["training"]["token_count_range"]
    result = {"features": {name: float(value) for name, value in features.iloc[0].items()},
              "advertencia": length_warning(int(features["token_count"].iloc[0]), low, high)}
    for target in TARGETS:
        contributions = bundle["models"][target].predict(features, pred_contrib=True)[0]
        result[target] = {"nota": float(contributions.sum()), "base": float(contributions[-1]),
                          "contribuciones": dict(zip(bundle["feature_columns"], map(float, contributions[:-1])))}
    return result


def copied_segments(text: str, prompt_text: str, n: int = 3) -> list[dict]:
    """Divide el resumen en tramos copiados del texto fuente (n-gramas que aparecen en él) y tramos propios.

    Usa la misma tokenización que las features de traslape, así que lo marcado es lo que mide
    ``trigramas_prompt_overlap`` con ``n=3``. Los tramos, unidos, reproducen el texto completo.
    """
    text = unicodedata.normalize("NFC", text)
    matches = list(TOKEN_PATTERN.finditer(text.replace("’", "'").lower()))
    tokens = [match.group() for match in matches]
    source_ngrams = set(generate_ngrams(tokenize_text(prompt_text), n))
    copied = [False] * len(tokens)
    for start, ngram in enumerate(generate_ngrams(tokens, n)):
        if ngram in source_ngrams:
            copied[start:start + n] = [True] * n

    segments: list[dict] = []
    position = 0
    for i, match in enumerate(matches):
        # Un tramo copiado se extiende sobre los espacios y signos entre dos palabras copiadas seguidas
        end = matches[i + 1].start() if copied[i] and i + 1 < len(matches) and copied[i + 1] else match.end()
        if match.start() > position:
            segments.append({"texto": text[position:match.start()], "copiado": False})
        segments.append({"texto": text[match.start():end], "copiado": copied[i]})
        position = end
    if position < len(text):
        segments.append({"texto": text[position:], "copiado": False})

    merged: list[dict] = []
    for segment in segments:
        if merged and merged[-1]["copiado"] == segment["copiado"]:
            merged[-1]["texto"] += segment["texto"]
        else:
            merged.append(dict(segment))
    return merged


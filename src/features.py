"""Features manuales del resumen, independientes del tema del texto fuente."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ngrams import prompt_ngram_overlap
from src.tokenization import tokenize_text


def manual_features(summaries: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    """Calcula largo, diversidad léxica y traslape de n-gramas con el texto fuente.

    Devuelve una fila por resumen, con el mismo índice que ``summaries``.
    """
    summary_tokens = summaries["text"].map(tokenize_text)
    prompt_tokens = prompts.set_index("prompt_id")["prompt_text"].map(tokenize_text).to_dict()

    features = pd.DataFrame(index=summaries.index)
    features["token_count"] = summary_tokens.str.len()
    features["log_token_count"] = np.log(features["token_count"])
    features["unique_token_count"] = summary_tokens.map(lambda tokens: len(set(tokens)))
    features["lexical_diversity"] = features["unique_token_count"] / features["token_count"]
    for n, label in [(1, "palabras"), (2, "bigramas"), (3, "trigramas")]:
        features[f"{label}_prompt_overlap"] = [
            prompt_ngram_overlap(tokens, prompt_tokens[prompt_id], n=n)
            for tokens, prompt_id in zip(summary_tokens, summaries["prompt_id"])
        ]
    return features

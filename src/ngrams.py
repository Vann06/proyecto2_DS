"""Funciones reutilizables para el análisis exploratorio de palabras y n-gramas."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import pandas as pd


def generate_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Devuelve las secuencias contiguas de longitud ``n`` de una lista de tokens."""
    if n < 1:
        raise ValueError("n debe ser un entero mayor o igual que 1.")
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def ngram_frequency_table(
    token_documents: Iterable[Sequence[str]],
    n: int = 1,
) -> pd.DataFrame:
    """Resume frecuencia absoluta, relativa y prevalencia documental de n-gramas."""
    documents = list(token_documents)
    occurrence_counts: Counter[tuple[str, ...]] = Counter()
    document_counts: Counter[tuple[str, ...]] = Counter()

    for tokens in documents:
        document_ngrams = generate_ngrams(tokens, n)
        occurrence_counts.update(document_ngrams)
        document_counts.update(set(document_ngrams))

    total_occurrences = sum(occurrence_counts.values())
    document_total = len(documents)
    rows = [
        {
            "ngram": " ".join(ngram),
            "frecuencia_absoluta": count,
            "frecuencia_relativa": count / total_occurrences if total_occurrences else 0.0,
            "documentos": document_counts[ngram],
            "prevalencia_documental": (
                document_counts[ngram] / document_total if document_total else 0.0
            ),
        }
        for ngram, count in occurrence_counts.items()
    ]
    columns = [
        "ngram",
        "frecuencia_absoluta",
        "frecuencia_relativa",
        "documentos",
        "prevalencia_documental",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["frecuencia_absoluta", "documentos", "ngram"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def performance_groups_by_prompt(
    frame: pd.DataFrame,
    target: str,
    prompt_column: str = "prompt_id",
) -> tuple[pd.Series, pd.DataFrame]:
    """Forma grupos bajo, medio y alto usando terciles calculados dentro de cada prompt."""
    required = {prompt_column, target}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {sorted(missing)}")
    if frame[target].isna().any():
        raise ValueError(f"{target} contiene valores faltantes.")

    groups = pd.Series(index=frame.index, dtype="object", name=f"nivel_{target}")
    threshold_rows = []
    for prompt_id, prompt_frame in frame.groupby(prompt_column, sort=True):
        lower, upper = prompt_frame[target].quantile([1 / 3, 2 / 3])
        if lower >= upper:
            raise ValueError(
                f"No se pueden formar tres niveles distintos de {target} para {prompt_id}."
            )
        prompt_groups = pd.Series("medio", index=prompt_frame.index, dtype="object")
        prompt_groups.loc[prompt_frame[target] <= lower] = "bajo"
        prompt_groups.loc[prompt_frame[target] >= upper] = "alto"
        groups.loc[prompt_frame.index] = prompt_groups
        counts = prompt_groups.value_counts()
        threshold_rows.append(
            {
                prompt_column: prompt_id,
                "objetivo": target,
                "tercil_inferior": lower,
                "tercil_superior": upper,
                "n_bajo": int(counts.get("bajo", 0)),
                "n_medio": int(counts.get("medio", 0)),
                "n_alto": int(counts.get("alto", 0)),
            }
        )
    return groups, pd.DataFrame(threshold_rows)


def compare_extreme_group_prevalence(
    frame: pd.DataFrame,
    tokens_column: str,
    group_column: str,
    n: int = 1,
    prompt_column: str = "prompt_id",
    min_document_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara prevalencia y frecuencia relativa entre niveles alto y bajo.

    Sólo considera n-gramas con al menos ``min_document_count`` documentos en el
    corpus. La frecuencia relativa se calcula primero por documento para controlar
    su longitud. La tabla agregada promedia las diferencias dentro de cada prompt,
    evitando que el prompt con más respuestas domine el resultado.
    """
    required = {tokens_column, group_column, prompt_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {sorted(missing)}")

    global_table = ngram_frequency_table(frame[tokens_column], n=n)
    candidates = set(
        global_table.loc[
            global_table["documentos"] >= min_document_count, "ngram"
        ]
    )
    prompts = sorted(frame[prompt_column].unique())
    rows = []
    for prompt_id in prompts:
        prompt_frame = frame.loc[frame[prompt_column] == prompt_id]
        for level in ("bajo", "alto"):
            documents = prompt_frame.loc[
                prompt_frame[group_column] == level, tokens_column
            ].tolist()
            document_counts: Counter[str] = Counter()
            relative_frequency_sums: Counter[str] = Counter()
            for tokens in documents:
                observed = [" ".join(value) for value in generate_ngrams(tokens, n)]
                observed_counts = Counter(observed)
                document_counts.update(set(observed_counts).intersection(candidates))
                if observed:
                    relative_frequency_sums.update(
                        {
                            ngram: count / len(observed)
                            for ngram, count in observed_counts.items()
                            if ngram in candidates
                        }
                    )
            for ngram in candidates:
                count = document_counts[ngram]
                rows.append(
                    {
                        prompt_column: prompt_id,
                        "nivel": level,
                        "ngram": ngram,
                        "documentos_grupo": len(documents),
                        "documentos_con_ngram": count,
                        "prevalencia": count / len(documents) if documents else 0.0,
                        "frecuencia_relativa_media": (
                            relative_frequency_sums[ngram] / len(documents)
                            if documents
                            else 0.0
                        ),
                    }
                )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    prevalence = detail.pivot(
        index=[prompt_column, "ngram"],
        columns="nivel",
        values=["prevalencia", "frecuencia_relativa_media"],
    ).fillna(0.0)
    prevalence.columns = [f"{metric}_{level}" for metric, level in prevalence.columns]
    prevalence = prevalence.reset_index()
    for metric in ("prevalencia", "frecuencia_relativa_media"):
        for level in ("bajo", "alto"):
            column = f"{metric}_{level}"
            if column not in prevalence:
                prevalence[column] = 0.0
    prevalence["diferencia_prevalencia"] = (
        prevalence["prevalencia_alto"] - prevalence["prevalencia_bajo"]
    )
    prevalence["diferencia_frecuencia_relativa"] = (
        prevalence["frecuencia_relativa_media_alto"]
        - prevalence["frecuencia_relativa_media_bajo"]
    )

    coverage = (
        detail.groupby([prompt_column, "ngram"], as_index=False)["documentos_con_ngram"]
        .sum()
        .assign(presente=lambda value: value["documentos_con_ngram"] > 0)
        .groupby("ngram", as_index=False)["presente"]
        .sum()
        .rename(columns={"presente": "prompts_con_ngram"})
    )
    aggregate = (
        prevalence.groupby("ngram", as_index=False)
        .agg(
            prevalencia_baja=("prevalencia_bajo", "mean"),
            prevalencia_alta=("prevalencia_alto", "mean"),
            diferencia_prevalencia_media=("diferencia_prevalencia", "mean"),
            frecuencia_relativa_baja=("frecuencia_relativa_media_bajo", "mean"),
            frecuencia_relativa_alta=("frecuencia_relativa_media_alto", "mean"),
            diferencia_frecuencia_relativa=(
                "diferencia_frecuencia_relativa",
                "mean",
            ),
            prompts_diferencia_positiva=(
                "diferencia_frecuencia_relativa",
                lambda values: int((values > 0).sum()),
            ),
            prompts_diferencia_negativa=(
                "diferencia_frecuencia_relativa",
                lambda values: int((values < 0).sum()),
            ),
        )
        .merge(coverage, on="ngram", how="left", validate="one_to_one")
        .merge(
            global_table[["ngram", "frecuencia_absoluta", "documentos"]],
            on="ngram",
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            ["diferencia_frecuencia_relativa", "documentos"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
    return prevalence, aggregate


def prompt_ngram_overlap(
    summary_tokens: Sequence[str],
    prompt_tokens: Sequence[str],
    n: int = 1,
) -> float:
    """Calcula la proporción de n-gramas únicos del resumen presentes en su prompt."""
    summary_ngrams = set(generate_ngrams(summary_tokens, n))
    if not summary_ngrams:
        return 0.0
    prompt_ngrams = set(generate_ngrams(prompt_tokens, n))
    return len(summary_ngrams.intersection(prompt_ngrams)) / len(summary_ngrams)

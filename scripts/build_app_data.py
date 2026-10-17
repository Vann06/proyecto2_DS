"""Junta en app/data/ los datos que usa la aplicación: resúmenes con features, predicciones y resultados de modelos.

Lee los archivos locales de data/interim/, data/processed/ y outputs/tables/ (generados por los notebooks 01 a 12),
así que se ejecuta una vez en una computadora que los tenga. Los archivos de salida sí se versionan.
No guarda el texto de los resúmenes: las gráficas solo necesitan sus notas y features.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_csv_files
from src.evaluation import regression_metrics
from src.fusion import FEATURE_COLUMNS, TARGETS, build_features

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
TABLES_DIR = PROJECT_ROOT / "outputs/tables"

# Etapa de cada modelo: "final" es la versión que entra a la tabla del notebook 12
FINAL, BEFORE_TUNING, DISCARDED = "final", "antes del ajuste", "variante descartada"

# nombre en los archivos de predicciones: (nombre en la aplicación, familia, notebook, etapa)
OOF_MODELS: dict[str, tuple[str, str, str, str]] = {
    "Dummy (media)": ("Dummy (media)", "Referencia", "08", FINAL),
    "Solo largo + Ridge": ("Solo largo + Ridge", "Referencia", "08", FINAL),
    "A. Features + Ridge": ("Features manuales + Ridge", "Features", "08", FINAL),
    "A. Features + LightGBM": ("Features manuales + LightGBM", "Features", "08", FINAL),
    "B. Embedding crudo + Ridge": ("Embedding crudo + Ridge", "Embeddings", "08", FINAL),
    "B. Embedding sin tema + Ridge": ("Embedding sin tema + Ridge", "Embeddings", "08", FINAL),
    "B. Embedding sin tema + PCA(16) + Ridge": ("Embedding sin tema + PCA(16) + Ridge", "Embeddings", "08", FINAL),
    "Opción 1: Fusión temprana + Ridge": (
        "Fusión temprana (features + embedding PCA) + Ridge", "Fusión", "08", FINAL),
    "Opción 1: Fusión temprana + LightGBM": (
        "Fusión temprana (features + embedding) + LightGBM", "Fusión", "08", FINAL),
    "Opción 2: Stacking (LightGBM A + Ridge B)": (
        "Stacking (LightGBM features + Ridge embedding)", "Fusión", "08", FINAL),
    "Opción 1: Fusión mínima (A + cos_sim) + LightGBM": (
        "Fusión mínima + LightGBM (parámetros de 08)", "Fusión", "08", BEFORE_TUNING),
    "Fusión mínima + LightGBM ajustado (validación anidada)": (
        "Fusión mínima (features + cos_sim) + LightGBM", "Fusión", "08 → 11", FINAL),
    "DeBERTa · base (09)": ("DeBERTa-v3-base (fine-tuning)", "Transformer", "09 → 10", FINAL),
    "DeBERTa · 2 épocas": ("DeBERTa · 2 épocas", "Transformer", "10", DISCARDED),
    "DeBERTa · lr 1e-5": ("DeBERTa · lr 1e-5", "Transformer", "10", DISCARDED),
    "DeBERTa · + texto fuente": ("DeBERTa · + texto fuente", "Transformer", "10", DISCARDED),
    "Promedio 50/50: DeBERTa + fusión ajustada": (
        "Promedio 50/50: DeBERTa + fusión mínima", "Ensamble", "09–11", FINAL),
}
APP_MODEL: str = "Fusión mínima (features + cos_sim) + LightGBM"
BEST_MODEL: str = "Promedio 50/50: DeBERTa + fusión mínima"


def load_predictions(summary_order: pd.DataFrame) -> pd.DataFrame:
    """Predicciones out-of-fold en formato largo, con los nombres de la aplicación y el orden de los resúmenes."""
    oof = pd.concat([pd.read_csv(PROCESSED_DIR / name) for name in
                     ["08_oof_predictions.csv", "10_deberta_oof_predictions.csv", "11_oof_predictions.csv"]],
                    ignore_index=True)
    oof = oof[oof["modelo"].isin(OOF_MODELS)]
    oof["modelo"] = oof["modelo"].map(lambda name: OOF_MODELS[name][0])

    # Promedio 50/50 antes del ajuste: DeBERTa + fusión con los parámetros de 08 (lo calcula el notebook 11)
    wide = oof.pivot(index="student_id", columns="modelo", values=["pred_content", "pred_wording"])
    blend = pd.DataFrame({f"pred_{t}": 0.5 * wide[(f"pred_{t}", "DeBERTa-v3-base (fine-tuning)")]
                          + 0.5 * wide[(f"pred_{t}", "Fusión mínima + LightGBM (parámetros de 08)")]
                          for t in TARGETS}).reset_index()
    blend["modelo"] = "Promedio 50/50 (fusión con parámetros de 08)"

    predictions = pd.concat([oof[["student_id", "modelo", "pred_content", "pred_wording"]], blend],
                            ignore_index=True)
    counts = predictions.groupby("modelo")["student_id"].nunique()
    assert (counts == len(summary_order)).all() and not predictions.duplicated(["student_id", "modelo"]).any()
    order = {student_id: i for i, student_id in enumerate(summary_order["student_id"])}
    return (predictions.assign(_order=predictions["student_id"].map(order))
            .sort_values(["modelo", "_order"]).drop(columns="_order").reset_index(drop=True))


def model_tables(summaries: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas globales y por prompt de cada modelo: de sus predicciones o, para 07, de sus tablas de resultados."""
    catalog = {name: (family, notebook, stage) for name, family, notebook, stage in OOF_MODELS.values()}
    catalog["Promedio 50/50 (fusión con parámetros de 08)"] = ("Ensamble", "09–11", BEFORE_TUNING)
    y = summaries[list(TARGETS)].to_numpy()
    groups = summaries["prompt_id"].to_numpy()
    titles = summaries.drop_duplicates("prompt_id").set_index("prompt_id")["prompt_title"]

    global_rows, prompt_rows = [], []
    for name, rows in predictions.groupby("modelo", sort=False):
        pred = rows[["pred_content", "pred_wording"]].to_numpy()
        by_prompt = []
        for prompt_id in titles.index:
            mask = groups == prompt_id
            metrics = regression_metrics(y[mask], pred[mask])
            by_prompt.append(metrics["mcrmse"])
            prompt_rows.append({"modelo": name, "prompt_id": prompt_id, "prompt_title": titles[prompt_id],
                                **{key: metrics[key] for key in ["rmse_content", "rmse_wording", "mcrmse"]}})
        metrics = regression_metrics(y, pred)
        global_rows.append({"modelo": name, **{key: metrics[key] for key in
                                               ["rmse_content", "r2_content", "rmse_wording", "r2_wording", "mcrmse"]},
                            "desviacion_entre_prompts": float(np.std(by_prompt, ddof=1)),
                            "tiene_predicciones": True})

    lexical = pd.read_csv(TABLES_DIR / "07_tfidf_grouped_results.csv")
    folds = pd.read_csv(TABLES_DIR / "07_tfidf_grouped_folds.csv")
    for record in lexical[lexical["modelo"] != "Dummy media"].to_dict("records"):
        catalog[record["modelo"]] = ("Léxicos", "07", FINAL)
        global_rows.append({"modelo": record["modelo"],
                            **{key: record[key] for key in
                               ["rmse_content", "r2_content", "rmse_wording", "r2_wording", "mcrmse"]},
                            "desviacion_entre_prompts": record["desviacion_mcrmse_entre_prompts"],
                            "tiene_predicciones": False})
        for fold in folds[folds["modelo"] == record["modelo"]].to_dict("records"):
            prompt_rows.append({"modelo": record["modelo"], "prompt_id": fold["prompt_id"],
                                "prompt_title": fold["prompt_title"],
                                **{key: fold[key] for key in ["rmse_content", "rmse_wording", "mcrmse"]}})

    models = pd.DataFrame(global_rows)
    models.insert(1, "familia", models["modelo"].map(lambda name: catalog[name][0]))
    models.insert(2, "notebook", models["modelo"].map(lambda name: catalog[name][1]))
    models.insert(3, "etapa", models["modelo"].map(lambda name: catalog[name][2]))
    models["es_modelo_aplicacion"] = models["modelo"] == APP_MODEL
    models["es_mejor_modelo"] = models["modelo"] == BEST_MODEL
    return models.sort_values("mcrmse").reset_index(drop=True), pd.DataFrame(prompt_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "app/data")
    args = parser.parse_args()

    datasets = load_csv_files(PROJECT_ROOT / "data/interim", ["summaries_train_clean.csv", "prompts_train_clean.csv"])
    summaries = datasets["summaries_train_clean"]
    prompts = datasets["prompts_train_clean"]
    summaries = summaries.merge(prompts[["prompt_id", "prompt_title"]], on="prompt_id", validate="many_to_one")
    features = build_features(summaries, prompts)
    summaries = pd.concat([summaries[["student_id", "prompt_id", "prompt_title", *TARGETS]], features], axis=1)

    predictions = load_predictions(summaries)
    models, models_by_prompt = model_tables(summaries, predictions)

    # Las cifras deben coincidir con la tabla final del notebook 12
    final = pd.read_csv(TABLES_DIR / "12_final_comparison.csv")
    check = models[models["etapa"] == FINAL].merge(final[["modelo", "mcrmse"]], on="modelo", suffixes=("", "_12"))
    assert len(check) == len(final) and np.allclose(check["mcrmse"], check["mcrmse_12"]), "No coincide con el 12."

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries[["student_id", "prompt_id", "prompt_title", *TARGETS, *FEATURE_COLUMNS]].to_parquet(
        args.output_dir / "resumenes.parquet", index=False)
    predictions.to_parquet(args.output_dir / "predicciones.parquet", index=False)
    models.to_csv(args.output_dir / "modelos.csv", index=False)
    models_by_prompt.to_csv(args.output_dir / "modelos_por_prompt.csv", index=False)
    prompts[["prompt_id", "prompt_title", "prompt_question", "prompt_text"]].to_csv(
        args.output_dir / "prompts.csv", index=False)
    # Historial de Optuna del notebook 11: una fila por prueba de cada búsqueda
    pd.read_csv(TABLES_DIR / "11_optuna_trials.csv").to_csv(args.output_dir / "optuna_pruebas.csv", index=False)

    print(f"Resúmenes: {len(summaries)} | modelos: {len(models)} "
          f"({models['tiene_predicciones'].sum()} con predicciones por resumen)")
    for path in sorted(args.output_dir.iterdir()):
        print(f"  {path.name}: {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

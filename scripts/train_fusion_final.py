"""Entrena la fusión mínima con todos los resúmenes y la guarda para la aplicación."""

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_csv_files
from src.evaluation import regression_metrics
from src.fusion import TARGETS, build_features, train_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/interim")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/fusion_lightgbm.joblib")
    args = parser.parse_args()

    datasets = load_csv_files(args.data_dir, ["summaries_train_clean.csv", "prompts_train_clean.csv"])
    summaries = datasets["summaries_train_clean"]
    prompts = datasets["prompts_train_clean"]
    print(f"Resúmenes: {len(summaries)} | prompts: {len(prompts)}")

    features = build_features(summaries, prompts)
    y = summaries[list(TARGETS)].to_numpy()
    bundle = train_fusion(features, y)
    bundle["training"]["prompts"] = prompts["prompt_title"].tolist()

    # Error sobre los mismos datos de entrenamiento: solo comprueba que el ajuste funcionó, no es una estimación
    # del error con prompts nuevos (esa es la del notebook 11, con validación por prompt).
    fitted = features.assign(**{t: bundle["models"][t].predict(features) for t in TARGETS})[list(TARGETS)]
    print(f"MCRMSE en entrenamiento (optimista): {regression_metrics(y, fitted.to_numpy())['mcrmse']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    print(f"Modelo guardado en {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

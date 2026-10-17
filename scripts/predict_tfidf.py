"""Predicción de content y wording para un resumen y su texto fuente."""

import argparse
import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tfidf import predict_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "data/processed/07_tfidf_model.joblib")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("No se encontró el modelo. Ejecute notebooks/07_tfidf_analysis.ipynb.")
    try:
        summary = args.summary_file.read_text(encoding="utf-8")
        prompt = args.prompt_file.read_text(encoding="utf-8")
        result = predict_scores(joblib.load(args.model), summary, prompt)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Predicción de content y wording con la fusión mínima.

Un resumen:   python scripts/predict_fusion.py --summary-file resumen.txt --prompt-file fuente.txt
Varios (CSV): python scripts/predict_fusion.py --summaries-csv resumenes.csv --prompts-csv prompts.csv --output pred.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion import TARGETS, predict_frame, predict_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--summaries-csv", type=Path, help="Columnas: prompt_id, text (y cualquier otra, p. ej. student_id).")
    parser.add_argument("--prompts-csv", type=Path, help="Columnas: prompt_id, prompt_text.")
    parser.add_argument("--output", type=Path, help="CSV de salida en el modo de varios resúmenes.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/fusion_lightgbm.joblib")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("No se encontró el modelo. Ejecute scripts/train_fusion_final.py.")
    single = args.summary_file and args.prompt_file
    batch = args.summaries_csv and args.prompts_csv and args.output
    if not (single or batch):
        parser.error("Use --summary-file y --prompt-file, o --summaries-csv, --prompts-csv y --output.")

    try:
        bundle = joblib.load(args.model)
        if single:
            result = predict_scores(bundle, args.summary_file.read_text(encoding="utf-8"),
                                    args.prompt_file.read_text(encoding="utf-8"))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            summaries = pd.read_csv(args.summaries_csv)
            predictions = predict_frame(bundle, summaries, pd.read_csv(args.prompts_csv))
            predictions = predictions.rename(columns={target: f"pred_{target}" for target in TARGETS})
            summaries.join(predictions).to_csv(args.output, index=False)
            print(f"{len(summaries)} predicciones guardadas en {args.output}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

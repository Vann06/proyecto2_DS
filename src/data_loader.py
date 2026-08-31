"""Utilidades centralizadas para localizar y cargar los datos del proyecto."""

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_TRAIN_FILES = ("summaries_train.csv", "prompts_train.csv")
OPTIONAL_COMPETITION_FILES = (
    "summaries_test.csv",
    "prompts_test.csv",
    "sample_submission.csv",
)


def find_project_root(start: str | Path | None = None) -> Path:
    """Encuentra la raíz del repositorio desde la raíz o desde ``notebooks/``."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError(
        "No se encontró la raíz del proyecto. Ejecute el notebook dentro del repositorio."
    )


def missing_files(data_path: str | Path, filenames: Iterable[str]) -> list[str]:
    """Devuelve los nombres de archivos que no existen en ``data_path``."""
    base_path = Path(data_path)
    return [name for name in filenames if not (base_path / name).is_file()]


def load_csv_files(
    data_path: str | Path,
    filenames: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Carga varios CSV y los indexa por el nombre del archivo sin extensión."""
    base_path = Path(data_path)
    missing = missing_files(base_path, filenames)
    if missing:
        raise FileNotFoundError(
            "Faltan archivos requeridos en "
            f"{base_path}: {', '.join(missing)}"
        )

    return {
        Path(filename).stem: pd.read_csv(base_path / filename)
        for filename in filenames
    }


def load_training_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los resúmenes y prompts de entrenamiento."""
    datasets = load_csv_files(data_path, REQUIRED_TRAIN_FILES)
    return datasets["summaries_train"], datasets["prompts_train"]


def load_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alias retrocompatible de :func:`load_training_data`."""
    return load_training_data(data_path)

"""Lectura y validación de los CSV que suben los usuarios."""

from __future__ import annotations

import csv
import io

import pandas as pd

MAX_FILE_BYTES = 10 * 1024 * 1024

# Columna estándar: (nombres aceptados en el archivo, obligatoria)
SUMMARY_COLUMNS: dict[str, tuple[tuple[str, ...], bool]] = {
    "id": (("id", "student_id"), False),
    "prompt_id": (("prompt_id",), True),
    "text": (("text", "resumen"), True),
}
PROMPT_COLUMNS: dict[str, tuple[tuple[str, ...], bool]] = {
    "prompt_id": (("prompt_id",), True),
    "prompt_text": (("prompt_text", "texto_fuente"), True),
    "prompt_title": (("prompt_title", "titulo"), False),
    "prompt_question": (("prompt_question", "consigna"), False),
}

TEMPLATES: dict[str, str] = {
    "resumenes.csv": (
        "id,prompt_id,text\n"
        "est-001,fotosintesis,\"Photosynthesis is how plants make food using sunlight. Chlorophyll captures the light "
        "and the plant turns carbon dioxide and water into glucose, releasing oxygen.\"\n"
        "est-002,39c16e,\"Aristotle says an ideal tragedy makes the audience feel pity and fear, and the hero falls "
        "because of a mistake, not because he is evil.\"\n"
    ),
    "prompts.csv": (
        "prompt_id,prompt_title,prompt_question,prompt_text\n"
        "fotosintesis,Photosynthesis,Summarize how plants make their food.,\"Photosynthesis is the process by which "
        "green plants use sunlight to make their own food. Chlorophyll captures light energy, and the plant turns "
        "carbon dioxide and water into glucose. Oxygen is released as a by-product.\"\n"
    ),
}


class UploadError(ValueError):
    """Error en un archivo subido, con un mensaje que explica cómo corregirlo."""


def read_upload(content: bytes, name: str, spec: dict[str, tuple[tuple[str, ...], bool]]) -> pd.DataFrame:
    """Lee un CSV, encuentra sus columnas (aceptando nombres alternativos) y devuelve las columnas estándar.

    Conserva las demás columnas del archivo, para devolverlas junto con los scores.
    """
    if len(content) > MAX_FILE_BYTES:
        raise UploadError(f"El {name} pesa más de {MAX_FILE_BYTES // (1024 * 1024)} MB.")
    if not content.strip():
        raise UploadError(f"El {name} está vacío.")
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        # sep=None detecta si el separador es coma o punto y coma (Excel en español usa punto y coma)
        frame = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, csv.Error) as error:
        raise UploadError(f"El {name} no es un CSV válido: {error}") from error

    frame.columns = [column.strip() for column in frame.columns]
    lower = {column.lower(): column for column in frame.columns}
    renames, missing = {}, []
    for standard, (aliases, required) in spec.items():
        found = next((lower[alias] for alias in aliases if alias in lower), None)
        if found:
            renames[found] = standard
        elif required:
            missing.append(" o ".join(aliases))
    if missing:
        raise UploadError(f"Al {name} le faltan columnas: {', '.join(missing)}. "
                          f"Columnas encontradas: {', '.join(frame.columns) or 'ninguna'}.")
    frame = frame.rename(columns=renames)
    for standard in spec:
        if standard in frame:
            frame[standard] = frame[standard].str.strip()
    frame = frame[~(frame[[name for name, (_, required) in spec.items() if required]] == "").all(axis=1)]
    if frame.empty:
        raise UploadError(f"El {name} no tiene filas con datos.")

    for standard, (_, required) in spec.items():
        if required and (frame[standard] == "").any():
            rows = (frame.index[frame[standard] == ""] + 2).tolist()[:10]
            raise UploadError(f"En el {name}, la columna {standard} está vacía en las filas {rows}.")
    return frame.reset_index(drop=True)

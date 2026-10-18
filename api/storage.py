"""Prompts que guardan los usuarios, en una base SQLite (un archivo, sin servidor aparte)."""

from __future__ import annotations

import re
import secrets
import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROMPT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def slugify(text: str) -> str:
    """Id legible a partir de un título: «The Water Cycle» → «the-water-cycle»."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:48] or "prompt"


class PromptStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    prompt_id TEXT PRIMARY KEY,
                    prompt_title TEXT NOT NULL,
                    prompt_question TEXT NOT NULL DEFAULT '',
                    prompt_text TEXT NOT NULL,
                    creado TEXT NOT NULL
                )""")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Una conexión por operación, porque FastAPI atiende peticiones en varios hilos. Se confirma y se cierra
        # siempre: «with sqlite3.connect()» solo confirma la transacción y deja la conexión abierta.
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def list(self) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM prompts ORDER BY creado")]

    def get(self, prompt_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM prompts WHERE prompt_id = ?", (prompt_id,)).fetchone()
        return dict(row) if row else None

    def new_id(self, title: str, taken: set[str]) -> str:
        """Id a partir del título; si ya existe, le agrega un sufijo al azar."""
        base = slugify(title)
        candidate = base
        while candidate in taken or self.get(candidate):
            candidate = f"{base}-{secrets.token_hex(2)}"
        return candidate

    def create(self, prompt_id: str, title: str, question: str, text: str) -> dict:
        """Guarda un prompt nuevo. Lanza ``KeyError`` si el id ya existe."""
        row = {"prompt_id": prompt_id, "prompt_title": title, "prompt_question": question, "prompt_text": text,
               "creado": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            with self._connect() as connection:
                connection.execute("INSERT INTO prompts VALUES (:prompt_id, :prompt_title, :prompt_question, "
                                   ":prompt_text, :creado)", row)
        except sqlite3.IntegrityError as error:
            raise KeyError(prompt_id) from error
        return row

    def delete(self, prompt_id: str) -> bool:
        with self._connect() as connection:
            return connection.execute("DELETE FROM prompts WHERE prompt_id = ?", (prompt_id,)).rowcount > 0

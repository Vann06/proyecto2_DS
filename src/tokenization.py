"""Tokenización de los textos de CommonLit sin modificar su contenido original."""

import re
import unicodedata


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    """Separa palabras y números, conserva contracciones y pasa a minúsculas."""
    if not isinstance(text, str):
        raise TypeError("El texto debe ser una cadena sin valores faltantes.")
    normalized = unicodedata.normalize("NFC", text).replace("’", "'").lower()
    return TOKEN_PATTERN.findall(normalized)

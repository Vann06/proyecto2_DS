"""Ejecuta un notebook y guarda sus resultados en el mismo archivo."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    path = args.notebook.resolve()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=args.timeout, kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print(f"Notebook ejecutado: {path.name}")


if __name__ == "__main__":
    main()

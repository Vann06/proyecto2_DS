"""Ejecuta uno o varios notebooks, en el orden dado, y guarda sus resultados en el mismo archivo."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", type=Path, nargs="+")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    # La consola de Windows (cp1252) no imprime algunos caracteres de los notebooks, como «→».
    sys.stdout.reconfigure(errors="replace")
    root = Path(__file__).resolve().parents[1]
    for notebook_path in args.notebooks:
        run_notebook(notebook_path.resolve(), root, args.timeout)


def run_notebook(path, root, timeout):
    notebook = nbformat.read(path, as_version=4)
    def report_progress(cell, cell_index, **kwargs):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                print(output.get("text", "").strip()[:1200], flush=True)

    with tempfile.TemporaryDirectory(prefix="commonlit_ipython_") as profile:
        old_profile = os.environ.get("IPYTHONDIR")
        os.environ["IPYTHONDIR"] = profile
        client = NotebookClient(
            notebook, timeout=timeout, kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
            on_cell_executed=report_progress,
        )
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
            if old_profile is None:
                os.environ.pop("IPYTHONDIR", None)
            else:
                os.environ["IPYTHONDIR"] = old_profile
    print(f"Notebook ejecutado: {path.name}")


if __name__ == "__main__":
    main()

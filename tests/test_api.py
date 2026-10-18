"""Pruebas de la API: prompts guardados, calificación por lote y por archivo.

Cargan el modelo y los datos reales de app/data/, así que tardan unos segundos en arrancar.
"""

import csv
import io
import os
import tempfile
import unittest
from pathlib import Path

TEMP_DIR = tempfile.TemporaryDirectory()
os.environ["GLOSA_DB"] = str(Path(TEMP_DIR.name) / "pruebas.db")   # antes de importar la API

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

SOURCE = ("Photosynthesis is the process by which green plants use sunlight to make their own food. Chlorophyll "
          "captures light energy, and the plant turns carbon dioxide and water into glucose. Oxygen is released.")
SUMMARY = ("Photosynthesis is how plants make food using sunlight. Chlorophyll captures the light and the plant turns "
           "carbon dioxide and water into glucose, which it uses to grow, and it releases oxygen into the air.")


def csv_bytes(rows: list[dict], delimiter: str = ",") -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_saved_prompt_lifecycle(self):
        created = self.client.post("/api/prompts", json={"prompt_title": "The Water Cycle", "prompt_text": SOURCE})
        self.assertEqual(created.status_code, 201)
        prompt_id = created.json()["prompt_id"]
        self.assertTrue(prompt_id.startswith("the-water-cycle"))

        listed = {row["prompt_id"]: row for row in self.client.get("/api/prompts").json()}
        self.assertEqual(listed[prompt_id]["origen"], "usuario")
        self.assertEqual(listed["39c16e"]["origen"], "commonlit")

        graded = self.client.post("/api/calificar", json={"text": SUMMARY, "prompt_id": prompt_id})
        self.assertEqual(graded.status_code, 200)
        self.assertIn("content", graded.json())

        self.assertEqual(self.client.delete(f"/api/prompts/{prompt_id}").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/prompts/{prompt_id}").status_code, 404)

    def test_prompt_ids_cannot_collide(self):
        body = {"prompt_id": "repetido", "prompt_title": "A", "prompt_text": SOURCE}
        self.assertEqual(self.client.post("/api/prompts", json=body).status_code, 201)
        self.assertEqual(self.client.post("/api/prompts", json=body).status_code, 409)
        commonlit = {"prompt_id": "39c16e", "prompt_title": "B", "prompt_text": SOURCE}
        self.assertEqual(self.client.post("/api/prompts", json=commonlit).status_code, 409)
        self.assertEqual(self.client.delete("/api/prompts/39c16e").status_code, 403)

    def test_batch_mixes_new_and_known_prompts(self):
        body = {"prompts": [{"prompt_id": "foto", "prompt_text": SOURCE}],
                "resumenes": [{"id": "a", "prompt_id": "foto", "text": SUMMARY},
                              {"id": "b", "prompt_id": "39c16e", "text": "A tragedy should cause pity and fear."}]}
        result = self.client.post("/api/calificar/lote", json=body).json()
        self.assertEqual([row["id"] for row in result["resumenes"]], ["a", "b"])
        self.assertIsNotNone(result["resumenes"][1]["advertencia"])   # 7 palabras: fuera del rango
        self.assertEqual({row["prompt_id"] for row in result["resumen_por_prompt"]}, {"foto", "39c16e"})

    def test_batch_rejects_unknown_prompt_and_repeated_ids(self):
        unknown = {"resumenes": [{"id": "a", "prompt_id": "nada", "text": SUMMARY}]}
        response = self.client.post("/api/calificar/lote", json=unknown)
        self.assertEqual(response.status_code, 422)
        self.assertIn("No existe el prompt «nada»", response.json()["detail"])
        repeated = {"resumenes": [{"id": "a", "prompt_id": "39c16e", "text": SUMMARY}] * 2}
        self.assertEqual(self.client.post("/api/calificar/lote", json=repeated).status_code, 422)

    def test_file_upload_returns_scores_and_saves_prompts(self):
        summaries = csv_bytes([{"student_id": "e1", "prompt_id": "sol", "text": SUMMARY, "grupo": "A"},
                               {"student_id": "e2", "prompt_id": "39c16e", "text": SUMMARY, "grupo": "B"}],
                              delimiter=";")
        prompts = csv_bytes([{"prompt_id": "sol", "titulo": "Sol", "prompt_text": SOURCE}])
        response = self.client.post("/api/calificar/archivo", data={"guardar_prompts": "true"},
                                    files={"resumenes": ("r.csv", summaries), "prompts": ("p.csv", prompts)})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["prompts_guardados"], ["sol"])
        self.assertEqual(self.client.get("/api/prompts").json()[-1]["prompt_title"], "Sol")

        # El prompt ya quedó guardado: el segundo archivo no necesita traerlo, y puede pedir el CSV de vuelta
        again = self.client.post("/api/calificar/archivo?formato=csv", files={"resumenes": ("r.csv", summaries)})
        rows = list(csv.DictReader(io.StringIO(again.content.decode("utf-8-sig"))))
        self.assertEqual([row["id"] for row in rows], ["e1", "e2"])
        self.assertEqual(rows[0]["grupo"], "A")   # conserva las columnas originales
        self.assertIn("percentil_wording", rows[0])

    def test_file_upload_explains_missing_columns(self):
        bad = csv_bytes([{"prompt": "39c16e", "resumen": SUMMARY}])
        response = self.client.post("/api/calificar/archivo", files={"resumenes": ("r.csv", bad)})
        self.assertEqual(response.status_code, 422)
        self.assertIn("prompt_id", response.json()["detail"])

    def test_templates_are_valid_uploads(self):
        summaries = self.client.get("/api/plantillas/resumenes.csv").content
        prompts = self.client.get("/api/plantillas/prompts.csv").content
        response = self.client.post("/api/calificar/archivo",
                                    files={"resumenes": ("r.csv", summaries), "prompts": ("p.csv", prompts)})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["resumenes"]), 2)


if __name__ == "__main__":
    unittest.main()

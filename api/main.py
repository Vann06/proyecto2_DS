"""API de la aplicación: califica resúmenes con la fusión mínima y entrega los datos de las gráficas.

Desarrollo:  uvicorn api.main:app --reload            (y en web/: npm run dev)
Producción:  cd web && npm run build, luego uvicorn api.main:app  (sirve también el frontend compilado)
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.storage import PROMPT_ID_PATTERN, PromptStore
from api.uploads import PROMPT_COLUMNS, SUMMARY_COLUMNS, TEMPLATES, UploadError, read_upload
from src.fusion import (FEATURE_COLUMNS, FINAL_PARAMS, FIXED_PARAMS, TARGETS, copied_segments, explain_scores,
                        load_nlp, predict_frame)
from src.tokenization import tokenize_text

DATA_DIR = PROJECT_ROOT / "app" / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "fusion_lightgbm.joblib"
WEB_DIST = PROJECT_ROOT / "web" / "dist"
# Base de los prompts guardados por los usuarios; GLOSA_DB permite usar otra (por ejemplo, en las pruebas)
DB_PATH = Path(os.environ.get("GLOSA_DB", PROJECT_ROOT / "app" / "storage" / "glosa.db"))

# Nombre y explicación de cada feature, para la interfaz
FEATURES: dict[str, dict[str, str]] = {
    "token_count": {"nombre": "Número de palabras", "detalle": "Cuántas palabras tiene el resumen."},
    "log_token_count": {"nombre": "Largo en escala logarítmica",
                        "detalle": "El mismo largo, pero pasar de 20 a 40 palabras pesa más que de 200 a 220."},
    "unique_token_count": {"nombre": "Palabras distintas", "detalle": "Cuántas palabras diferentes usa."},
    "lexical_diversity": {"nombre": "Diversidad léxica",
                          "detalle": "Palabras distintas entre palabras totales: qué tanto repite."},
    "palabras_prompt_overlap": {"nombre": "Palabras del texto fuente",
                                "detalle": "Parte de sus palabras que también aparece en el texto fuente."},
    "bigramas_prompt_overlap": {"nombre": "Pares de palabras del texto fuente",
                                "detalle": "Parte de sus pares de palabras seguidas que aparece en el texto fuente."},
    "trigramas_prompt_overlap": {"nombre": "Frases copiadas del texto fuente",
                                 "detalle": "Parte de sus secuencias de tres palabras que aparece tal cual en el "
                                            "texto fuente."},
    "cos_sim": {"nombre": "Similitud con el texto fuente",
                "detalle": "Qué tan parecido es el tema del resumen al del texto, según los vectores de spaCy."},
}
# Parámetros de 08, el punto de partida del ajuste del notebook 11
PARAMS_08: dict = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=0.0, reg_alpha=0.0)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Todo se carga una vez al arrancar; cada petición solo lee de memoria
    state["bundle"] = joblib.load(MODEL_PATH)
    state["nlp"] = load_nlp()
    state["resumenes"] = pd.read_parquet(DATA_DIR / "resumenes.parquet")
    state["predicciones"] = pd.read_parquet(DATA_DIR / "predicciones.parquet")
    state["modelos"] = pd.read_csv(DATA_DIR / "modelos.csv")
    state["modelos_por_prompt"] = pd.read_csv(DATA_DIR / "modelos_por_prompt.csv")
    state["prompts"] = pd.read_csv(DATA_DIR / "prompts.csv")
    state["optuna"] = pd.read_csv(DATA_DIR / "optuna_pruebas.csv")
    state["store"] = PromptStore(DB_PATH)
    yield
    state.clear()


app = FastAPI(title="Premerie", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2048)


def columns(frame: pd.DataFrame, decimals: int = 4) -> dict[str, list]:
    """Tabla en formato de columnas ({columna: [valores]}), más compacto en JSON que una lista de filas."""
    return {name: (frame[name].round(decimals) if frame[name].dtype.kind == "f" else frame[name]).tolist()
            for name in frame.columns}


def records(frame: pd.DataFrame) -> list[dict]:
    """Filas como diccionarios, con NaN convertido en null."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


@app.get("/api/meta")
def meta():
    models = state["modelos"]
    app_model = models[models["es_modelo_aplicacion"]].iloc[0]
    best_model = models[models["es_mejor_modelo"]].iloc[0]
    return {
        "features": [{"id": name, **FEATURES[name]} for name in FEATURE_COLUMNS],
        "objetivos": list(TARGETS),
        "modelo_aplicacion": {"nombre": app_model["modelo"], "mcrmse": app_model["mcrmse"]},
        "mejor_modelo": {"nombre": best_model["modelo"], "mcrmse": best_model["mcrmse"]},
        "rango_palabras": state["bundle"]["training"]["token_count_range"],
        "resumenes": len(state["resumenes"]),
    }


def all_prompts() -> pd.DataFrame:
    """Los 4 textos de CommonLit y los que guardaron los usuarios, con su origen."""
    saved = pd.DataFrame(state["store"].list(),
                         columns=["prompt_id", "prompt_title", "prompt_question", "prompt_text", "creado"])
    return pd.concat([state["prompts"].assign(origen="commonlit"), saved.assign(origen="usuario")],
                     ignore_index=True)


@app.get("/api/prompts")
def prompts():
    """Todos los prompts. Los de CommonLit traen cuántos resúmenes de entrenamiento tienen y su nota media."""
    summary = state["resumenes"].groupby("prompt_id").agg(
        resumenes=("content", "size"), content_media=("content", "mean"), wording_media=("wording", "mean"))
    return records(all_prompts().merge(summary.reset_index(), on="prompt_id", how="left"))


class NewPrompt(BaseModel):
    prompt_title: str = Field(min_length=1, max_length=200)
    prompt_text: str = Field(min_length=1, max_length=50_000)
    prompt_question: str = Field(default="", max_length=1_000)
    prompt_id: str | None = Field(default=None, pattern=PROMPT_ID_PATTERN.pattern,
                                  description="Opcional; si no se indica, se crea a partir del título.")


def save_prompt(prompt_id: str | None, title: str, question: str, text: str) -> dict:
    """Guarda un prompt; el id no puede repetir uno de CommonLit ni uno guardado."""
    commonlit = set(state["prompts"]["prompt_id"])
    if not prompt_id:
        prompt_id = state["store"].new_id(title, taken=commonlit)
    if not PROMPT_ID_PATTERN.match(prompt_id):
        raise HTTPException(422, f"El id «{prompt_id}» solo puede tener letras, números, guiones y guiones bajos.")
    if prompt_id in commonlit:
        raise HTTPException(409, f"«{prompt_id}» es uno de los textos de CommonLit; use otro id.")
    if not tokenize_text(text):
        raise HTTPException(422, "El texto fuente debe tener al menos una palabra.")
    try:
        return state["store"].create(prompt_id, title.strip(), question.strip(), text.strip())
    except KeyError:
        raise HTTPException(409, f"Ya existe un prompt guardado con el id «{prompt_id}».") from None


@app.post("/api/prompts", status_code=201)
def create_prompt(prompt: NewPrompt):
    return save_prompt(prompt.prompt_id, prompt.prompt_title, prompt.prompt_question, prompt.prompt_text)


@app.delete("/api/prompts/{prompt_id}", status_code=204)
def delete_prompt(prompt_id: str):
    if prompt_id in set(state["prompts"]["prompt_id"]):
        raise HTTPException(403, "Los textos de CommonLit no se pueden borrar.")
    if not state["store"].delete(prompt_id):
        raise HTTPException(404, f"No existe un prompt guardado con el id «{prompt_id}».")


@app.get("/api/resumenes")
def resumenes():
    return columns(state["resumenes"].drop(columns="student_id"))


@app.get("/api/modelos")
def modelos():
    return {"modelos": records(state["modelos"]), "por_prompt": records(state["modelos_por_prompt"])}


@app.get("/api/predicciones/{modelo}")
def predicciones(modelo: str):
    """Predicciones out-of-fold de un modelo, en el mismo orden de filas que /api/resumenes."""
    rows = state["predicciones"][state["predicciones"]["modelo"] == modelo]
    if rows.empty:
        raise HTTPException(404, f"No hay predicciones por resumen para el modelo «{modelo}».")
    return columns(rows[["pred_content", "pred_wording"]])


@app.get("/api/optuna")
def optuna_trials():
    return records(state["optuna"])


class GradeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    prompt_id: str | None = None
    prompt_text: str | None = Field(default=None, max_length=50_000)

    @model_validator(mode="after")
    def one_source(self):
        if bool(self.prompt_id) == bool(self.prompt_text):
            raise ValueError("Indique un prompt_id conocido o el texto fuente (prompt_text), no ambos.")
        return self


@app.post("/api/calificar")
def calificar(request: GradeRequest):
    if request.prompt_id:
        known = all_prompts().set_index("prompt_id")
        if request.prompt_id not in known.index:
            raise HTTPException(404, f"No existe el prompt «{request.prompt_id}».")
        source = known.loc[request.prompt_id, "prompt_text"]
        reference = state["resumenes"][state["resumenes"]["prompt_id"] == request.prompt_id]
        if reference.empty:   # prompt guardado por un usuario: se compara con todos los resúmenes
            reference = state["resumenes"]
    else:
        source = request.prompt_text
        reference = state["resumenes"]
    try:
        result = explain_scores(state["bundle"], request.text, source, state["nlp"])
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

    # Percentil: qué parte de los resúmenes de referencia tiene una nota menor que la predicha
    for target in TARGETS:
        result[target]["percentil"] = float(100 * (reference[target] < result[target]["nota"]).mean())
    result["referencia"] = {"resumenes": len(reference), "prompt_id": request.prompt_id,
                            **{target: reference[target].round(4).tolist() for target in TARGETS}}
    result["rangos_tipicos"] = {name: {"p10": float(reference[name].quantile(0.1)),
                                       "mediana": float(reference[name].median()),
                                       "p90": float(reference[name].quantile(0.9))} for name in FEATURE_COLUMNS}
    result["tramos"] = copied_segments(request.text, source)
    return result


def quoted(values: list) -> str:
    """«a», «b» y «c»: para listar ids en los mensajes de error."""
    items = [f"«{value}»" for value in values]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def score_summaries(summaries: pd.DataFrame, extra_prompts: pd.DataFrame) -> pd.DataFrame:
    """Scores de varios resúmenes (columnas id, prompt_id, text).

    El texto fuente de cada resumen se busca primero en ``extra_prompts`` (los que vienen en la petición o en el
    archivo), después en los guardados y por último en los de CommonLit.
    """
    if summaries["id"].duplicated().any():
        repeated = summaries.loc[summaries["id"].duplicated(), "id"].unique()[:10].tolist()
        raise HTTPException(422, f"Cada resumen debe tener un id distinto. Repetidos: {quoted(repeated)}.")
    if extra_prompts["prompt_id"].duplicated().any():
        raise HTTPException(422, "Cada prompt_id debe aparecer una sola vez en los prompts.")
    known = all_prompts()[["prompt_id", "prompt_text"]]
    prompts = pd.concat([extra_prompts[["prompt_id", "prompt_text"]],
                         known[~known["prompt_id"].isin(extra_prompts["prompt_id"])]], ignore_index=True)
    unknown = sorted(set(summaries["prompt_id"]).difference(prompts["prompt_id"]))
    if unknown:
        noun = "el prompt" if len(unknown) == 1 else "los prompts"
        raise HTTPException(422, f"No existe {noun} {quoted(unknown[:10])}: agrégalo en «Agregar prompt» o súbelo "
                                 "en el archivo de prompts.")
    used = prompts[prompts["prompt_id"].isin(summaries["prompt_id"])]
    try:
        predictions = predict_frame(state["bundle"], summaries[["prompt_id", "text"]], used, state["nlp"])
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

    # Percentil frente a los resúmenes de su texto si es uno de los 4 de CommonLit (y no se reemplazó); si no, frente
    # a los 7,165 resúmenes de entrenamiento
    reference = state["resumenes"]
    own_reference = set(reference["prompt_id"]) - set(extra_prompts["prompt_id"])
    sorted_scores = {key: {target: np.sort(rows[target].to_numpy()) for target in TARGETS}
                     for key, rows in [("todos", reference), *reference.groupby("prompt_id")]}
    result = summaries[["id", "prompt_id"]].copy()
    for target in TARGETS:
        result[target] = predictions[target].to_numpy()
        pools = [prompt_id if prompt_id in own_reference else "todos" for prompt_id in result["prompt_id"]]
        # searchsorted cuenta cuántas notas de referencia son menores que la predicha
        result[f"percentil_{target}"] = [
            100 * np.searchsorted(sorted_scores[pool][target], value) / len(sorted_scores[pool][target])
            for pool, value in zip(pools, result[target])]
    result["advertencia"] = predictions["advertencia"].to_numpy()
    return result


def batch_response(result: pd.DataFrame) -> dict:
    by_prompt = result.groupby("prompt_id", as_index=False, sort=False).agg(
        resumenes=("id", "size"), content_media=("content", "mean"), wording_media=("wording", "mean"),
        con_advertencia=("advertencia", "count"))
    return {"resumenes": records(result), "resumen_por_prompt": records(by_prompt)}


MAX_BATCH = 5000


class BatchPrompt(BaseModel):
    prompt_id: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1, max_length=50_000)


class BatchSummary(BaseModel):
    id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=20_000)


class BatchRequest(BaseModel):
    prompts: list[BatchPrompt] = Field(default_factory=list)
    resumenes: list[BatchSummary] = Field(min_length=1, max_length=MAX_BATCH)


@app.post("/api/calificar/lote")
def calificar_lote(request: BatchRequest):
    """Califica varios resúmenes de una vez (JSON).

    Cada resumen indica su ``prompt_id``: uno de CommonLit, uno guardado o uno de los que vienen en ``prompts``
    dentro de la misma petición (un texto nuevo se envía una sola vez aunque tenga muchos resúmenes).
    """
    summaries = pd.DataFrame([summary.model_dump() for summary in request.resumenes])
    extra = pd.DataFrame([prompt.model_dump() for prompt in request.prompts], columns=["prompt_id", "prompt_text"])
    return batch_response(score_summaries(summaries, extra))


@app.post("/api/calificar/archivo")
async def calificar_archivo(
    resumenes: UploadFile = File(description="CSV con las columnas id (opcional), prompt_id y text."),
    prompts: UploadFile | None = File(default=None, description="CSV opcional con prompt_id y prompt_text "
                                                                "(y opcionalmente prompt_title y prompt_question)."),
    guardar_prompts: bool = Form(default=False, description="Guardar los prompts del archivo para usarlos después."),
    formato: str = Query(default="json", pattern="^(json|csv)$",
                         description="json: scores en JSON; csv: el archivo original con los scores agregados."),
):
    """Califica los resúmenes de un archivo CSV. /api/plantillas/resumenes.csv muestra el formato."""
    try:
        summaries = read_upload(await resumenes.read(), "archivo de resúmenes", SUMMARY_COLUMNS)
        extra = (read_upload(await prompts.read(), "archivo de prompts", PROMPT_COLUMNS) if prompts
                 else pd.DataFrame(columns=["prompt_id", "prompt_text"]))
    except UploadError as error:
        raise HTTPException(422, str(error)) from error
    if len(summaries) > MAX_BATCH:
        raise HTTPException(413, f"El archivo tiene {len(summaries)} resúmenes; el máximo es {MAX_BATCH}.")
    if "id" not in summaries:
        summaries.insert(0, "id", [f"fila-{i + 2}" for i in range(len(summaries))])   # número de fila del CSV

    result = score_summaries(summaries, extra)
    saved: list[str] = []
    if guardar_prompts:
        for row in extra.to_dict("records"):
            existing = state["store"].get(row["prompt_id"])
            if existing and existing["prompt_text"] == row["prompt_text"]:
                continue   # ya estaba guardado igual
            save_prompt(row["prompt_id"], row.get("prompt_title") or row["prompt_id"], row.get("prompt_question", ""),
                        row["prompt_text"])
            saved.append(row["prompt_id"])

    if formato == "csv":
        output = summaries.merge(result.drop(columns="prompt_id"), on="id", how="left", validate="one_to_one")
        return Response(output.to_csv(index=False).encode("utf-8-sig"), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="resumenes_calificados.csv"'})
    return batch_response(result) | {"prompts_guardados": saved}


@app.get("/api/plantillas/{nombre}")
def plantilla(nombre: str):
    """Archivos de ejemplo con la estructura que espera /api/calificar/archivo."""
    if nombre not in TEMPLATES:
        raise HTTPException(404, f"Las plantillas disponibles son: {', '.join(TEMPLATES)}.")
    return Response(TEMPLATES[nombre].encode("utf-8-sig"), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{nombre}"'})


class LabParams(BaseModel):
    # Mismos rangos que la búsqueda de Optuna del notebook 11
    n_estimators: int = Field(ge=50, le=1500)
    learning_rate: float = Field(ge=0.005, le=0.3)
    num_leaves: int = Field(ge=2, le=64)
    min_child_samples: int = Field(ge=5, le=200)
    subsample: float = Field(ge=0.5, le=1.0)
    colsample_bytree: float = Field(ge=0.4, le=1.0)
    reg_lambda: float = Field(ge=0.0, le=10.0)
    reg_alpha: float = Field(ge=0.0, le=10.0)


class LabRequest(BaseModel):
    objetivo: str = Field(pattern="^(content|wording)$")
    params: LabParams


def lopo_rmse(target: str, params: dict) -> dict:
    """Entrena con 3 prompts y mide el RMSE en el cuarto, para cada prompt (validación Leave-One-Prompt-Out)."""
    from lightgbm import LGBMRegressor

    data = state["resumenes"]
    X, y, groups = data[FEATURE_COLUMNS], data[target].to_numpy(), data["prompt_id"].to_numpy()
    predictions = np.zeros_like(y)
    by_prompt = []
    for prompt_id, title in data.drop_duplicates("prompt_id")[["prompt_id", "prompt_title"]].itertuples(index=False):
        valid = groups == prompt_id
        model = LGBMRegressor(**params, **FIXED_PARAMS).fit(X[~valid], y[~valid])
        predictions[valid] = model.predict(X[valid])
        by_prompt.append({"prompt_id": prompt_id, "prompt_title": title,
                          "rmse": float(np.sqrt(np.mean((y[valid] - predictions[valid]) ** 2)))})
    return {"rmse": float(np.sqrt(np.mean((y - predictions) ** 2))), "por_prompt": by_prompt}


@lru_cache(maxsize=8)
def reference_rmse(target: str, which: str) -> dict:
    return lopo_rmse(target, FINAL_PARAMS[target] if which == "ajustados" else PARAMS_08)


@app.get("/api/laboratorio/referencias/{objetivo}")
def lab_references(objetivo: str):
    if objetivo not in TARGETS:
        raise HTTPException(404, "El objetivo debe ser content o wording.")
    return {"parametros_ajustados": {key: value for key, value in FINAL_PARAMS[objetivo].items()
                                     if key != "subsample_freq"},
            "ajustados": reference_rmse(objetivo, "ajustados"), "parametros_08": reference_rmse(objetivo, "08")}


@app.post("/api/laboratorio")
def laboratorio(request: LabRequest):
    return lopo_rmse(request.objetivo, request.params.model_dump() | {"subsample_freq": 1})


# Frontend compilado (web/dist): cualquier ruta que no sea de la API devuelve la aplicación de React
if WEB_DIST.is_dir():
    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        file = (WEB_DIST / path).resolve()
        if path and file.is_file() and WEB_DIST.resolve() in file.parents:
            # Los archivos de assets/ llevan un hash en el nombre: si cambian, cambia el nombre, así que se pueden
            # guardar en caché para siempre
            immutable = path.startswith("assets/")
            return FileResponse(file, headers={"Cache-Control": "public, max-age=31536000, immutable"} if immutable else None)
        # index.html se revisa en cada visita, para que después de un «npm run build» se cargue la versión nueva
        return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})

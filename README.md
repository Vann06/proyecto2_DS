# CommonLit - Evaluate Student Summaries

Proyecto 2 - Resultados y Análisis
CC3084 Data Science
Universidad del Valle de Guatemala

**Premary** es la aplicación del proyecto: califica el `content` y el `wording` de resúmenes
escritos por estudiantes y muestra cómo se comparan los modelos que entrenamos.

## Levantar la aplicación

### Requisitos

- Python 3.12
- Node.js 22 o superior (incluye npm)

### Primera vez

Desde la raíz del repositorio:

```bash
# 1. Entorno de Python y dependencias (incluye el modelo en_core_web_md de spaCy, unos 40 MB)
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell o cmd)
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. Dependencias del frontend y build
cd web
npm install
npm run build
cd ..
```

El modelo (`models/fusion_lightgbm.joblib`) y los datos de las gráficas (`app/data/`) ya vienen
en el repositorio: no hace falta ejecutar los notebooks para usar la aplicación.

### Ejecutar

```bash
uvicorn api.main:app --port 8000
```

La aplicación queda en **http://127.0.0.1:8000** y la documentación interactiva de la API en
http://127.0.0.1:8000/docs. La API tarda unos segundos en arrancar porque carga el modelo y spaCy.

### Desarrollo del frontend

Para ver los cambios del frontend al instante, sin volver a compilar, se usan dos terminales:

```bash
uvicorn api.main:app --port 8000      # terminal 1: la API
npm --prefix web run dev              # terminal 2: el frontend en http://localhost:5173
```

Vite reenvía las llamadas a `/api` hacia el puerto 8000. Si se cambia algo en `api/` o `src/`,
hay que reiniciar uvicorn (en Windows, `--reload` no siempre detecta los cambios).

### Pruebas

```bash
python -W ignore -m unittest discover -s tests
```

Prueban las features, el modelo empaquetado y la API (prompts guardados, calificación por lote
y por archivo).

### Problemas comunes

- **`address already in use` / `solo se permite un uso de cada dirección de socket`:** ya hay algo
  en el puerto 8000 (o 5173). Cierra la otra terminal o usa otro puerto con `--port`.
- **La página sale en blanco en http://127.0.0.1:8000:** falta el build; ejecuta `npm run build`
  dentro de `web/`.
- **`Can't find model 'en_core_web_md'`:** vuelve a ejecutar `pip install -r requirements.txt`
  con el entorno activado.

## La aplicación

| Página | Qué hace |
|---|---|
| **Inicio** | Califica un resumen y explica la nota: percentil frente a los resúmenes del mismo texto, qué medidas la subieron o bajaron y qué frases se copiaron del texto fuente. También califica archivos CSV con muchos resúmenes y devuelve un CSV con las notas. |
| **Agregar prompt** | Guarda textos fuente nuevos, uno por uno o desde un CSV, para calificar resúmenes sobre ellos. |
| **Cómo funciona** | Recorrido con scroll por el camino de un resumen: features manuales, similitud con embeddings, fusión y los dos LightGBM. |
| **Métricas** | Comparación de los 23 modelos del proyecto: MCRMSE, error por prompt, búsqueda de Optuna y, por modelo, predicho contra real. |

Las notas están en la escala de CommonLit, que no es de 0 a 10: 0 es un resumen promedio, y en
los datos `content` va de −1.7 a 3.9 y `wording` de −2.0 a 4.3. Por eso la aplicación muestra
también el percentil.

### API

| Método | Ruta | Uso |
|---|---|---|
| GET | `/api/prompts` | Los 4 textos de CommonLit y los que guardaron los usuarios (`origen`) |
| POST | `/api/prompts` | Guarda un texto fuente (`prompt_title`, `prompt_text`, `prompt_question` y `prompt_id` opcionales) |
| DELETE | `/api/prompts/{prompt_id}` | Borra un texto guardado (los de CommonLit no se pueden borrar) |
| POST | `/api/calificar` | Califica un resumen con todo el detalle: scores, percentiles, efecto de cada medida y frases copiadas |
| POST | `/api/calificar/lote` | Califica hasta 5,000 resúmenes enviados en JSON |
| POST | `/api/calificar/archivo` | Califica un CSV de resúmenes, con un CSV opcional de prompts; `?formato=csv` devuelve el archivo con los scores agregados |
| GET | `/api/plantillas/resumenes.csv`, `/api/plantillas/prompts.csv` | Plantillas con la estructura que esperan los archivos |
| GET | `/api/meta`, `/api/resumenes`, `/api/modelos`, `/api/predicciones/{modelo}`, `/api/optuna` | Datos de las gráficas |
| GET, POST | `/api/laboratorio/...` | Entrena LightGBM en vivo con los hiperparámetros elegidos |

**Archivos que se pueden subir** (CSV en UTF-8, separados por coma o punto y coma):

- Resúmenes: `prompt_id` y `text` (o `resumen`), y `id` o `student_id` opcional. Las demás
  columnas se conservan en el CSV de salida.
- Prompts: `prompt_id` y `prompt_text` (o `texto_fuente`), y `prompt_title` (o `titulo`) y
  `prompt_question` (o `consigna`) opcionales. Con `guardar_prompts=true` quedan guardados
  para calificar después solo con el `prompt_id`.

Los prompts guardados viven en `app/storage/glosa.db` (SQLite). Es local de cada computadora y no
se versiona.

### Qué se versiona y qué no

| Se versiona | No se versiona (se genera) |
|---|---|
| `models/fusion_lightgbm.joblib`, `app/data/` | `.venv/`, `web/node_modules/` (`pip install`, `npm install`) |
| `api/`, `src/`, `web/src/`, `web/public/` | `web/dist/` (`npm run build`) |
| Tablas y figuras finales de `outputs/` | `app/storage/` (prompts guardados en cada computadora) |
| | `data/raw/`, `data/interim/`, `data/processed/` (datos de Kaggle y salidas de los notebooks) |

## Descripción del proyecto

Este proyecto analiza el conjunto de datos CommonLit - Evaluate Student Summaries y compara
modelos de regresión para estimar `content` (qué tan bien captura el resumen las ideas del texto)
y `wording` (qué tan propio y claro es su lenguaje). La métrica es el MCRMSE, el promedio del
RMSE de las dos notas.

### Datos

Los datos provienen de la competencia CommonLit - Evaluate Student Summaries (Kaggle, 2023):
7,165 resúmenes sobre 4 textos fuente. Los archivos originales van en `data/raw/` y no se
versionan.

### Integrantes

- Angie Nadissa Vela López – 23764
- Vianka Vanessa Castro Ordoñez - 23201
- Ricardo Arturo Godínez Sánchez - 23247
- Ingrid Nina Alessandra Nájera Marakovits – 231088

### Estructura

| Carpeta | Contenido |
|---|---|
| `notebooks/` | Análisis y modelado, en orden numérico |
| `src/` | Código compartido por los notebooks, los scripts y la API |
| `scripts/` | Ejecutar notebooks, entrenar el modelo final, predecir y generar los datos de la app |
| `models/` | Modelo empaquetado para la aplicación |
| `app/data/` | Datos de las gráficas de la aplicación |
| `api/` | API en FastAPI |
| `web/` | Frontend en React + Vite |
| `tests/` | Pruebas |
| `outputs/` | Tablas y figuras de los notebooks |
| `docs/` | Informe, presentación y guías |

### Reproducir los notebooks

Para ejecutar los notebooks solo se necesitan estos archivos en `data/raw/`:

- `summaries_train.csv`
- `prompts_train.csv`

Los archivos de prueba y `sample_submission.csv` son opcionales. Los notebooks se ejecutan en
orden numérico. Los dos primeros cargan los datos, validan su calidad y generan copias limpias en
`data/interim/` sin modificar los originales.

```bash
jupyter notebook
# o, desde la raíz del proyecto, sin abrir Jupyter:
python scripts/run_notebook.py notebooks/06_bow_analysis.ipynb notebooks/07_tfidf_analysis.ipynb notebooks/08_features_embeddings.ipynb
```

Las tablas se guardan en `outputs/tables/` y las figuras en `outputs/figures/`.

### Flujo de los notebooks

Los notebooks forman un solo recorrido: cada uno abre con la pregunta que deja el anterior y
cierra con lo que aprendió y la pregunta que motiva el siguiente.

```
DATOS               00_data_loading → 01_data_quality
EDA                 02_eda_targets → 03_eda_text_features → 04_eda_prompts → 05_nlp_analysis
BASELINES LÉXICOS   06_bow_analysis → 07_tfidf_analysis
MODELADO            08_features_embeddings → 09_deberta_finetuning → 10_deberta_tuning (Colab, GPU)
AJUSTE              11_fusion_lightgbm_tuning (local, en paralelo con 10)
RESULTADO FINAL     12_comparacion_final
```

- **06 y 07** contienen los baselines léxicos (Dummy, BoW y TF-IDF). 07 los evalúa todos con
  Leave-One-Prompt-Out y búsqueda de hiperparámetros agrupada por prompt.
- **08** prueba features manuales, embeddings, fusión temprana y stacking con la misma
  validación. No vuelve a entrenar los modelos léxicos: lee
  `outputs/tables/07_tfidf_grouped_results.csv`, así que 07 debe ejecutarse antes. Requiere
  LightGBM y el modelo `en_core_web_md` de spaCy; la comparación con Sentence-BERT descarga
  `all-MiniLM-L6-v2`.
- **09** ajusta DeBERTa-v3-base (fine-tuning) con la misma validación. Necesita GPU, así que se
  ejecuta en Google Colab: el notebook detecta Colab y lee y escribe en
  `MyDrive/proyecto2_DS_deberta/`. Sus resultados se copian después a `data/processed/`
  (predicciones) y a `outputs/tables/` y `outputs/figures/`. Si la sesión se corta, al volver a
  ejecutarlo se saltan los folds terminados. Guía paso a paso:
  [`docs/ejecutar_09_en_colab.md`](docs/ejecutar_09_en_colab.md).
- **10** prueba tres variantes de DeBERTa fijadas de antemano (2 épocas, tasa de aprendizaje
  1e-5 y las oraciones del texto fuente más parecidas al resumen en la entrada). Las compara solas
  y en el promedio 50/50 con la fusión mínima de 08, y elige la variante con validación por
  prompt. También se ejecuta en Colab, en la misma carpeta de Drive; necesita las predicciones de
  08 y 09.
- **11** ajusta los hiperparámetros de la fusión mínima con LightGBM (Optuna) con validación
  anidada por prompt. Corre en local y necesita `optuna`.
- **12** reúne todos los modelos en una tabla de mejor a peor, cada uno en su versión final, en
  `outputs/tables/12_final_comparison.csv`, compara a los tres finalistas y separa dos
  decisiones: el mejor modelo es el promedio 50/50 de DeBERTa y la fusión mínima (MCRMSE 0.529),
  y el modelo para la aplicación es la fusión mínima con LightGBM (0.573), por ser liviana y
  fácil de empaquetar.
- 07, 08, 09, 10 y 11 guardan sus predicciones out-of-fold en `data/processed/` con el mismo
  formato, para la comparación final.

Se reportan dos evaluaciones distintas: una prueba estratificada para prompts conocidos y una
validación externa por prompt con ajuste interno agrupado. Las representaciones se comparan con
las mismas particiones.

El código compartido vive en `src/`: `tokenization.py` y `ngrams.py` (EDA), `bow.py` y `tfidf.py`
(representaciones léxicas), `features.py` (features manuales de 08), `evaluation.py` (métricas
comunes, incluido MCRMSE) y `fusion.py` (el modelo empaquetado para la aplicación).

El modelo TF-IDF completo y sus metadatos se generan en `data/processed/` y no se versionan. Para
predecir con él:

```bash
python scripts/predict_tfidf.py --summary-file resumen.txt --prompt-file fuente.txt
```

### Modelo de la aplicación

La aplicación usa la fusión mínima con LightGBM ajustado (notebook 12, sección 7). El modelo está
entrenado con los 7,165 resúmenes y guardado en `models/fusion_lightgbm.joblib` (unos 240 KB).
Ese archivo contiene los dos LightGBM (uno para `content` y otro para `wording`), el orden de las
features, los hiperparámetros, las versiones de las librerías y el rango de largo de los
resúmenes de entrenamiento. El modelo de spaCy no va dentro: se instala con `requirements.txt`.

Las 8 features son 7 manuales (largo, vocabulario y cuánto comparte el resumen con el texto
fuente en palabras, pares y tríos de palabras) y `cos_sim`, la similitud de tema entre el resumen
y el texto fuente con los vectores de spaCy.

Funciona con prompts y resúmenes nuevos sin reentrenar: solo necesita el resumen y el texto
fuente que se resumió.

```bash
# Un resumen
python scripts/predict_fusion.py --summary-file resumen.txt --prompt-file fuente.txt

# Varios resúmenes, de uno o varios prompts
#   resumenes.csv: prompt_id, text (y cualquier otra columna, p. ej. student_id)
#   prompts.csv:   prompt_id, prompt_text
python scripts/predict_fusion.py --summaries-csv resumenes.csv --prompts-csv prompts.csv --output predicciones.csv
```

Desde Python:

```python
import joblib
from src.fusion import predict_scores

bundle = joblib.load("models/fusion_lightgbm.joblib")   # cargar una sola vez
predict_scores(bundle, texto_del_resumen, texto_fuente)
# {"content": 0.03, "wording": 0.09, "advertencia": None}
```

`advertencia` avisa cuando el resumen es más corto o más largo que los de entrenamiento (25 a 660
palabras). Para volver a entrenar el modelo, por ejemplo con más datos, se ejecuta
`python scripts/train_fusion_final.py`, que lee `data/interim/`.

### Datos de la aplicación

`app/data/` reúne lo que necesitan las gráficas de la aplicación y se versiona para que funcione
en cualquier computadora:

| Archivo | Contenido |
|---|---|
| `resumenes.parquet` | Una fila por resumen: prompt, `content`, `wording` y las 8 features de la fusión mínima (sin el texto del resumen) |
| `predicciones.parquet` | Predicciones out-of-fold de los 18 modelos con predicciones por resumen (formato largo) |
| `modelos.csv` | Métricas globales de los 23 modelos, con familia, notebook y etapa (final, antes del ajuste o variante descartada) |
| `modelos_por_prompt.csv` | Las mismas métricas para cada prompt |
| `prompts.csv` | Título, pregunta y texto fuente de los 4 prompts |
| `optuna_pruebas.csv` | Las 600 pruebas de Optuna del notebook 11 (10 búsquedas de 60): hiperparámetros probados, RMSE y mejor RMSE hasta cada prueba |

Se generan con `python scripts/build_app_data.py`, que necesita los archivos locales de
`data/interim/` y `data/processed/` (los producen los notebooks 01 a 11) y comprueba que las
cifras coincidan con la tabla final del notebook 12.

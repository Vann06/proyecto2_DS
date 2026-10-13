# CommonLit - Evaluate Student Summaries

Proyecto 2 - Resultados y Análisis
CC3084 Data Science  
Universidad del Valle de Guatemala

## Descripción

Este proyecto analiza el conjunto de datos CommonLit - Evaluate Student
Summaries y compara modelos de regresión para estimar content y wording.

El objetivo es identificar patrones estadísticos y lingüísticos
asociados con las puntuaciones de `content` y `wording`
de los resúmenes escritos por estudiantes.

## Datos

Los datos provienen de la competencia:

CommonLit - Evaluate Student Summaries  
Kaggle, 2023.

Los archivos originales se almacenan en:

data/raw/

## Estructura

- `data/`: datasets
- `notebooks/`: análisis exploratorio
- `src/`: funciones reutilizables
- `outputs/`: gráficos y tablas
- `docs/`: informe y presentación

## Integrantes

- Angie Nadissa Vela López – 23764 
- Vianka Vanessa Castro Ordoñez - 23201 
- Ricardo Arturo Godínez Sánchez - 23247 
- Ingrid Nina Alessandra Nájera Marakovits – 231088 

## Ejecución

Crear/activar un entorno virtual e instalar dependencias:

```bash
pip install -r requirements.txt
```

Para ejecutar el EDA sólo se necesitan estos archivos en `data/raw/`:

- `summaries_train.csv`
- `prompts_train.csv`

Los archivos de prueba y `sample_submission.csv` son opcionales. Después,
ejecutar los notebooks en orden numérico. Los dos primeros realizan la carga,
validan la calidad y generan copias limpias en `data/interim/` sin modificar
los archivos originales.

```bash
jupyter notebook
```

Las tablas generadas por los análisis se guardan en `outputs/tables/`; las
figuras se guardarán en `outputs/figures/`.

Después de los notebooks 00 a 05, se ejecuta 06_bow_analysis.ipynb para
la representación BoW y 07_tfidf_analysis.ipynb para TF-IDF, similitud y
comparación de modelos. El notebook 06_modelado.ipynb contiene el análisis
de características manuales, embeddings y combinación de modelos.
Su ejecución requiere LightGBM y el modelo en_core_web_md de spaCy.
La comparación con Sentence-BERT requiere descargar all-MiniLM-L6-v2.

El análisis TF-IDF se puede ejecutar desde la raíz del proyecto con:

```bash
python scripts/run_notebook.py notebooks/07_tfidf_analysis.ipynb
```

Se reportan dos evaluaciones distintas: una prueba estratificada para
prompts conocidos y una validación externa por prompt con ajuste interno
agrupado. Las representaciones se comparan usando las mismas particiones.
Los archivos 07_tfidf de outputs/tables y outputs/figures se incluyen en
el repositorio para conservar los resultados de esta evaluación.

La redacción de la sección TF-IDF se conserva en docs/resultados_tfidf.txt.
El PDF con las tablas y figuras se genera con:

```bash
python scripts/render_tfidf_report.py
```

El modelo completo y sus metadatos se generan en data/processed y no se
versionan. Se utiliza el mismo vectorizador durante la predicción:

```bash
python scripts/predict_tfidf.py --summary-file resumen.txt --prompt-file fuente.txt
```

Los archivos de entrada contienen el resumen y su texto fuente en UTF-8.
Se producen las puntuaciones continuas de content y wording. Las pruebas
de vocabulario, similitud y persistencia se ejecutan con:

```bash
python -m unittest discover -s tests -v
```

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

### Flujo de los notebooks

Los notebooks forman un solo recorrido narrativo: cada uno abre con la pregunta
que deja el anterior y cierra con lo que aprendió y la pregunta que motiva el
siguiente.

```
DATOS               00_data_loading → 01_data_quality
EDA                 02_eda_targets → 03_eda_text_features → 04_eda_prompts → 05_nlp_analysis
BASELINES LÉXICOS   06_bow_analysis → 07_tfidf_analysis
MODELADO            08_features_embeddings → 09_deberta_finetuning → 10_deberta_tuning (Colab, GPU)
AJUSTE              11_fusion_lightgbm_tuning (local, en paralelo con 10)
RESULTADO FINAL     12_comparacion_final
```

- **06 y 07** contienen los baselines léxicos (Dummy, BoW y TF-IDF). 07 los
  evalúa todos con Leave-One-Prompt-Out y búsqueda de hiperparámetros
  agrupada por prompt.
- **08** prueba features manuales, embeddings, fusión temprana y stacking
  con la misma validación. No vuelve a entrenar los modelos léxicos: lee
  `outputs/tables/07_tfidf_grouped_results.csv`, así que 07 debe ejecutarse
  antes. Requiere LightGBM y el modelo `en_core_web_md` de spaCy; la
  comparación con Sentence-BERT descarga `all-MiniLM-L6-v2`.
- **09** ajusta DeBERTa-v3-base (fine-tuning) con la misma validación. Necesita
  GPU, así que se ejecuta en Google Colab: el notebook detecta Colab y lee y
  escribe en `MyDrive/proyecto2_DS_deberta/`. Sus resultados se copian después a
  `data/processed/` (predicciones) y `outputs/tables/` y `outputs/figures/` (tablas
  y figura). Si la sesión se corta, al volver a ejecutarlo se saltan los folds terminados.
  Guía paso a paso: [`docs/ejecutar_09_en_colab.md`](docs/ejecutar_09_en_colab.md).
- **10** prueba tres variantes de DeBERTa fijadas de antemano (2 épocas, tasa de
  aprendizaje 1e-5 y las oraciones del texto fuente más parecidas al resumen en la
  entrada). Las compara solas y en el
  promedio 50/50 con la fusión mínima de 08, y elige la variante con validación
  por prompt. También se ejecuta en Colab, en la misma carpeta de Drive; necesita
  las predicciones de 08 y 09.
- **11** ajusta los hiperparámetros de la fusión mínima con LightGBM (Optuna)
  con validación anidada por prompt. Corre en local y necesita `optuna`.
- **12** reúne todos los modelos en una tabla de mejor a peor, cada uno en su
  versión final (después del ajuste), en `outputs/tables/12_final_comparison.csv`.
- 07, 08, 09, 10 y 11 guardan sus predicciones out-of-fold en `data/processed/` con el
  mismo formato, para una comparación final posterior.

El código compartido vive en `src/`: `tokenization.py` y `ngrams.py` (EDA),
`bow.py` y `tfidf.py` (representaciones léxicas), `features.py` (features
manuales de 08) y `evaluation.py` (métricas comunes, incluido MCRMSE).

Los notebooks se pueden ejecutar en orden desde la raíz del proyecto con:

```bash
python scripts/run_notebook.py notebooks/06_bow_analysis.ipynb notebooks/07_tfidf_analysis.ipynb notebooks/08_features_embeddings.ipynb
```

Se reportan dos evaluaciones distintas: una prueba estratificada para
prompts conocidos y una validación externa por prompt con ajuste interno
agrupado. Las representaciones se comparan usando las mismas particiones.
Los archivos 07_tfidf de outputs/tables y outputs/figures se incluyen en
el repositorio para conservar los resultados de esta evaluación.

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

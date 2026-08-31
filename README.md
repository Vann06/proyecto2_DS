# CommonLit - Evaluate Student Summaries

Proyecto 2 - Análisis Exploratorio  
CC3084 Data Science  
Universidad del Valle de Guatemala

## Descripción

Este proyecto realiza un análisis exploratorio del conjunto de datos
CommonLit - Evaluate Student Summaries.

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

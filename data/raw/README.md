# Raw Data

Esta carpeta contiene los datos originales utilizados en el proyecto
**CommonLit - Evaluate Student Summaries**.

Los archivos provienen de la competencia de Kaggle:

CommonLit - Evaluate Student Summaries  
https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries

## Archivos esperados

Los datos originales de la competencia incluyen:

- `summaries_train.csv`
- `summaries_test.csv`
- `prompts_train.csv`
- `prompts_test.csv`
- `sample_submission.csv`

Los archivos originales no deben modificarse directamente.

Toda transformación debe guardarse posteriormente en:

- `data/interim/`
- `data/processed/`

---

## Descarga automática utilizando KaggleHub

Los datos pueden descargarse utilizando la librería `kagglehub`.

```python
import kagglehub

path = kagglehub.competition_download(
    "commonlit-evaluate-student-summaries"
)

print("Path to competition files:", path)
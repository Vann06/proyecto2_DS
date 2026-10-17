# Cómo ejecutar `09_deberta_finetuning.ipynb` en Google Colab

El notebook 09 entrena **DeBERTa-v3-base con ajuste fino** usando la misma validación que 07 y 08
(Leave-One-Prompt-Out, 4 folds). Necesita GPU, por eso se ejecuta en Google Colab.

El notebook detecta solo si está en Colab. En Colab lee y escribe en
`MyDrive/proyecto2_DS_deberta/`. En local usa `data/` y `outputs/` del repositorio.

---

## 1. Subir los datos a Google Drive

En **Mi unidad**, crear exactamente esta estructura:

```
MyDrive/proyecto2_DS_deberta/
└── data/
    ├── summaries_train_clean.csv     ← de data/interim/
    ├── prompts_train_clean.csv       ← de data/interim/
    └── 08_oof_predictions.csv        ← de data/processed/ (opcional, para comparar con 08)
```

- Los dos CSV limpios los genera `01_data_quality.ipynb`.
- `08_oof_predictions.csv` lo genera `08_features_embeddings.ipynb`. Sin él, el notebook se ejecuta igual, pero
  omite la comparación con 08 y el promedio 50/50.
- La carpeta `outputs/` la crea el notebook.

## 2. Abrir el notebook

1. Entrar a <https://colab.research.google.com>.
2. **Archivo → Subir cuaderno** y elegir `notebooks/09_deberta_finetuning.ipynb`.

> El repositorio es privado: abrirlo desde la pestaña GitHub de Colab requiere autorizar el acceso a
> repositorios privados. Subir el archivo es más simple.

## 3. Activar la GPU

**Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4 → Guardar.**

## 4. Prueba corta (unos 5 minutos)

1. En la celda de configuración (sección 1), cambiar:
   ```python
   SMOKE_TEST: bool = True
   ```
2. **Entorno de ejecución → Ejecutar todas.**
3. Aceptar el permiso de Google Drive cuando aparezca.
4. Revisar que:
   - aparezca `GPU: Tesla T4 | memoria: ~15 GB`;
   - el porcentaje de "Entradas que exceden 512 tokens" sea bajo;
   - aparezca `Fold 1 ... listo` sin errores, junto con una tabla de métricas.

La prueba usa 400 resúmenes, 1 fold y 1 época. Sus salidas van a `outputs/09_smoke_test/` y no se mezclan
con la corrida real. Sus métricas no significan nada; solo comprueban que todo funciona.

## 5. Corrida completa

1. Volver a poner:
   ```python
   SMOKE_TEST: bool = False
   ```
2. **Entorno de ejecución → Ejecutar todas.**
3. Duración medida en una T4: entre 7 y 9 minutos por fold, unos 35 minutos en total (puede variar según la
   GPU que asigne Colab).
4. Mantener la pestaña abierta, porque Colab gratis desconecta las sesiones inactivas.

### Si la sesión se corta

Cada fold guarda sus predicciones en Drive apenas termina
(`outputs/09_deberta_folds/09_deberta_fold_<prompt_id>.csv`). Para continuar:

1. Reconectar y revisar que siga activa la GPU T4.
2. **Ejecutar todas** otra vez.

Los folds terminados se cargan de Drive (`Fold N: ya existe`) y el entrenamiento sigue desde donde se quedó.

> Para empezar de cero (por ejemplo, después de cambiar un hiperparámetro), borrar la carpeta
> `outputs/09_deberta_folds/` y el archivo `outputs/09_deberta_fold_log.csv` en Drive.

## 6. Si algo falla

| Error | Solución |
|---|---|
| `CUDA out of memory` | Cambiar `batch_size=8` a `4` en `CONFIG` y luego **Entorno de ejecución → Reiniciar sesión → Ejecutar todas**. Si sigue fallando, bajar `max_len` a 384 |
| Error del tokenizador relacionado con `sentencepiece` | **Reiniciar sesión → Ejecutar todas**. La instalación ya se hizo y sobrevive al reinicio |
| El MSE de entrenamiento sale `nan` | Problema de fp16: en la celda de configuración poner `USE_AMP: bool = False` (más lento) |
| `FileNotFoundError` con los CSV | Revisar que los nombres y la carpeta `MyDrive/proyecto2_DS_deberta/data/` estén escritos exactamente igual |
| Sin GPU (`Sin GPU: el entrenamiento completo sería demasiado lento`) | Repetir el paso 3. Si Colab no asigna GPU, se agotó la cuota gratuita; esperar unas horas |

Si se cambia algún hiperparámetro, hay que anotarlo en la conclusión del notebook. Los valores de `CONFIG` se
fijaron antes de ver resultados.

## 7. Traer los resultados al repositorio

Desde `MyDrive/proyecto2_DS_deberta/outputs/`:

| Archivo | Destino en el repositorio |
|---|---|
| `09_deberta_oof_predictions.csv` | `data/processed/` |
| `09_deberta_results.csv` | `outputs/tables/` |
| `09_deberta_by_prompt.csv` | `outputs/tables/` |
| `09_deberta_fold_log.csv` | `outputs/tables/` |
| `09_deberta_comparison.csv` (si se subió `08_oof_predictions.csv`) | `outputs/tables/` |
| `09_deberta_pred_vs_real.png` | `outputs/figures/` |

Después:

1. Revisar que la celda de configuración diga `SMOKE_TEST: bool = False`.
2. **Archivo → Descargar → Descargar .ipynb** y reemplazar `notebooks/09_deberta_finetuning.ipynb`. Así el
   notebook queda versionado con sus resultados.
3. Completar la sección 7 (Conclusiones) del notebook con los números obtenidos.

Las tablas y la figura `09_*` están permitidas en `.gitignore`. `09_deberta_oof_predictions.csv` queda en
`data/processed/`, que no se versiona.

## Configuración usada

| Parámetro | Valor |
|---|---|
| Modelo | `microsoft/deberta-v3-base` |
| Entrada | `prompt_question [SEP] resumen`, máximo 512 tokens |
| Cabeza | Mean pooling + `Linear(768, 2)`, pérdida MSE |
| Épocas | 3, sin early stopping |
| Batch | 8 |
| Tasa de aprendizaje | 2e-5 (DeBERTa), 1e-3 (cabeza) |
| Weight decay | 0.01 |
| Warmup | 10% de los pasos, luego decaimiento lineal |
| Clipping de gradiente | 1.0 |
| Precisión | fp16 (mixta) en GPU |
| Semilla | 42 |
| Validación | `LeaveOneGroupOut` por `prompt_id` (mismos folds que 07 y 08) |

---

# Notebook 10: ajuste de DeBERTa (`10_deberta_tuning.ipynb`)

Se ejecuta igual que el 09, en la **misma carpeta de Drive**. No hay que subir datos nuevos: usa los CSV limpios y
`08_oof_predictions.csv` de `data/`, y `09_deberta_oof_predictions.csv`, que el 09 dejó en `outputs/`.

1. **Archivo → Subir cuaderno** → `notebooks/10_deberta_tuning.ipynb`, y moverlo a la carpeta
   `proyecto2_DS_deberta`.
2. Activar la **GPU T4**.
3. Opcional: prueba corta con `SMOKE_TEST: bool = True` (400 resúmenes, 2 folds y 1 época por variante; unos
   10 minutos). Sus salidas van a `outputs/10_smoke_test/`.
4. Corrida completa: `SMOKE_TEST: bool = False` → **Ejecutar todas**.

Duración estimada (no medida), a partir de los tiempos del 09:

| Variante | Por fold | Total |
|---|---|---|
| `dos_epocas` | ~5–6 min | ~25 min |
| `lr_1e-5` | ~8 min | ~35 min |
| `texto_fuente` | ~20–30 min (todas las entradas llegan a 512 tokens: pregunta, resumen y las oraciones del texto fuente más parecidas al resumen) | ~1.5–2 h |

El notebook imprime `tiempo estimado del fold` después de 50 lotes. Si no alcanza la sesión, se puede repartir: la
lista `VARIANTS_TO_RUN` de la celda de configuración indica qué variantes se entrenan, y lo terminado se carga de
Drive al volver a ejecutar.

Con `texto_fuente`, si aparece `CUDA out of memory`, cambiar en `VARIANTS` su `batch_size=4, grad_accum=2` por
`batch_size=2, grad_accum=4` y anotarlo en las conclusiones.

## Traer los resultados del 10 al repositorio

| Archivo en `outputs/` de Drive | Destino en el repositorio |
|---|---|
| `10_deberta_oof_predictions.csv` | `data/processed/` |
| `10_deberta_tuning_results.csv`, `10_deberta_tuning_by_prompt.csv`, `10_deberta_tuning_selection.csv`, `10_deberta_fold_log.csv` | `outputs/tables/` |
| `10_deberta_tuning_by_prompt.png` | `outputs/figures/` |

Después, descargar el `.ipynb` ejecutado (**Archivo → Descargar → Descargar .ipynb**) y reemplazar
`notebooks/10_deberta_tuning.ipynb`.

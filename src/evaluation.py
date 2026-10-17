"""Métricas de regresión compartidas por los notebooks de modelado."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


TARGETS = ("content", "wording")


def regression_metrics(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray,
    targets: Sequence[str] = TARGETS,
) -> dict[str, float]:
    """Calcula RMSE, MAE y R2 por objetivo, además de MCRMSE."""
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError("y_true y y_pred deben tener la misma forma.")
    if observed.ndim != 2 or observed.shape[1] != len(targets):
        raise ValueError("Las matrices deben tener una columna por objetivo.")

    metrics: dict[str, float] = {}
    rmse_values = []
    for index, target in enumerate(targets):
        rmse = root_mean_squared_error(observed[:, index], predicted[:, index])
        rmse_values.append(rmse)
        metrics[f"rmse_{target}"] = rmse
        metrics[f"mae_{target}"] = mean_absolute_error(
            observed[:, index], predicted[:, index]
        )
        metrics[f"r2_{target}"] = r2_score(observed[:, index], predicted[:, index])
    metrics["mcrmse"] = float(np.mean(rmse_values))
    return metrics

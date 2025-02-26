"""
Evaluation metrics and utilities for traffic flow prediction.
"""
import math
from typing import List, Union, Dict

import numpy as np
import sklearn.metrics as metrics


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value as a percentage
    """
    # Filter out zeros to avoid division by zero
    y_filtered = [y_true[i] for i in range(len(y_true)) if y_true[i] > 0]
    y_pred_filtered = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    if len(y_filtered) == 0:
        return 0.0

    # Calculate MAPE
    total_error = 0
    for i in range(len(y_filtered)):
        total_error += abs(y_filtered[i] - y_pred_filtered[i]) / y_filtered[i]

    mape = total_error * (100 / len(y_filtered))

    return mape


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate regression predictions with multiple metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of evaluation metrics
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = metrics.r2_score(y_true, y_pred)

    results = {
        "explained_variance": explained_variance,
        "mape": mape,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    return results

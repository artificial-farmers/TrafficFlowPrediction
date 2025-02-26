"""
Visualization utilities for traffic flow prediction.
"""
from typing import List, Tuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prediction_results(
    y_true: np.ndarray,
    y_preds: List[np.ndarray],
    model_names: List[str],
    start_date: str = "2016-3-4 00:00",
    freq: str = "5min",
    num_points: int = 288,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot prediction results against ground truth.

    Args:
        y_true: True values
        y_preds: List of predictions from different models
        model_names: Names of the models corresponding to predictions
        start_date: Start date for x-axis
        freq: Frequency of data points
        num_points: Number of points to plot
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)

    Returns:
        The created figure
    """
    # Create time index
    x = pd.date_range(start_date, periods=num_points, freq=freq)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Plot ground truth
    ax.plot(x, y_true[:num_points], label="True Data", linewidth=2)

    # Plot predictions
    for name, y_pred in zip(model_names, y_preds):
        ax.plot(x, y_pred[:num_points], label=name, linestyle="--", alpha=0.8)

    # Add legend and grid
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time of Day")
    plt.ylabel("Traffic Flow")
    plt.title("Traffic Flow Prediction Results")

    # Format x-axis with hours and minutes
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_history(
    history: dict,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training history.

    Args:
        history: Training history dictionary
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)

    Returns:
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot MAPE
    ax2.plot(history['mape'], label='Training MAPE')
    if 'val_mape' in history:
        ax2.plot(history['val_mape'], label='Validation MAPE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAPE (%)')
    ax2.set_title('Training and Validation MAPE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

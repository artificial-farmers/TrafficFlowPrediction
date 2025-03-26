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
    title: Optional[str] = None
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
        title: Custom title for the plot

    Returns:
        The created figure
    """
    # Ensure we don't exceed available data
    num_points = min(num_points, len(y_true))

    # Create time index
    x = pd.date_range(start_date, periods=num_points, freq=freq)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Plot ground truth
    ax.plot(x, y_true[:num_points], label="True Data", linewidth=2)

    # Plot predictions
    for name, y_pred in zip(model_names, y_preds):
        # Ensure we don't exceed available data
        pred_points = min(num_points, len(y_pred))
        ax.plot(x[:pred_points], y_pred[:pred_points], label=name, linestyle="--", alpha=0.8)

    # Add legend and grid
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time of Day")
    plt.ylabel("Traffic Flow")

    # Set title
    if title:
        plt.title(title)
    else:
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
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot training history.

    Args:
        history: Training history dictionary
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        title: Custom title for the plot

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

    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.85)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_site_comparison(
    df: pd.DataFrame,
    metric: str = 'mape',
    top_n: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot comparison of model performance across sites.

    Args:
        df: DataFrame with columns: site_id, model_type, and metrics
        metric: Metric to compare ('mape', 'rmse', 'r2', etc.)
        top_n: Number of top sites to display
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)

    Returns:
        The created figure
    """
    # Pivot the data for easier plotting
    pivot_df = df.pivot(index='site_id', columns='model_type', values=metric)

    # Sort based on the average metric value across all models
    if metric in ['r2', 'explained_variance']:  # Higher is better
        sorted_sites = pivot_df.mean(axis=1).sort_values(ascending=False).index[:top_n]
    else:  # Lower is better (mape, rmse, mae, mse)
        sorted_sites = pivot_df.mean(axis=1).sort_values(ascending=True).index[:top_n]

    # Filter for top sites
    plot_df = pivot_df.loc[sorted_sites]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the comparison
    plot_df.plot(kind='bar', ax=ax, width=0.8)

    # Add labels and title
    metric_name = {
        'mape': 'Mean Absolute Percentage Error (%)',
        'rmse': 'Root Mean Squared Error',
        'r2': 'R² Score',
        'mae': 'Mean Absolute Error',
        'mse': 'Mean Squared Error',
        'explained_variance': 'Explained Variance Score'
    }.get(metric, metric)

    plt.xlabel('SCATS Site ID')
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} Across Top {top_n} Sites')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title='Model')
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_performance_summary(
    df: pd.DataFrame,
    metrics: List[str] = ['mape', 'rmse', 'r2'],
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot summary of model performance with multiple metrics.

    Args:
        df: DataFrame with columns: model_type and metrics
        metrics: List of metrics to include
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)

    Returns:
        The created figure
    """
    # Number of metrics to plot
    n_metrics = len(metrics)

    # Create figure with subplots
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    # Metric names for labels
    metric_names = {
        'mape': 'Mean Absolute Percentage Error (%)',
        'rmse': 'Root Mean Squared Error',
        'r2': 'R² Score',
        'mae': 'Mean Absolute Error',
        'mse': 'Mean Squared Error',
        'explained_variance': 'Explained Variance Score'
    }

    # Plot each metric
    for i, metric in enumerate(metrics):
        # Determine if higher or lower is better
        if metric in ['r2', 'explained_variance']:
            ascending = False  # Higher is better
            best_label = 'Higher is better'
        else:
            ascending = True  # Lower is better
            best_label = 'Lower is better'

        # Sort data
        plot_data = df.sort_values(f'avg_{metric}', ascending=ascending)

        # Get colors for bars
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(plot_data)))

        # Create bar chart
        bars = axes[i].bar(
            plot_data['model_type'],
            plot_data[f'avg_{metric}'],
            color=colors,
            alpha=0.7
        )

        # Add error bars if min/max available
        if f'min_{metric}' in plot_data.columns and f'max_{metric}' in plot_data.columns:
            axes[i].errorbar(
                plot_data['model_type'],
                plot_data[f'avg_{metric}'],
                yerr=[
                    plot_data[f'avg_{metric}'] - plot_data[f'min_{metric}'],
                    plot_data[f'max_{metric}'] - plot_data[f'avg_{metric}']
                ],
                fmt='none',
                color='black',
                capsize=5
            )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width()/2.,
                height * 1.01,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        # Set labels and title
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel(metric_names.get(metric, metric))
        axes[i].set_title(f'Average {metric_names.get(metric, metric)} by Model ({best_label})')
        axes[i].grid(True, axis='y', alpha=0.3)

    # Add overall title
    plt.suptitle('Model Performance Summary Across All Sites', fontsize=16, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_traffic_patterns(
    df: pd.DataFrame,
    site_ids: List[str],
    date_column: str = 'Timestamp',
    flow_column: str = 'Traffic_Flow',
    group_by: str = 'hour',
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot traffic flow patterns for multiple sites.

    Args:
        df: DataFrame with traffic data
        site_ids: List of site IDs to include
        date_column: Name of date/time column
        flow_column: Name of traffic flow column
        group_by: How to group data ('hour', 'day', 'weekday', etc.)
        figsize: Figure size
        title: Custom title for the plot
        save_path: Path to save the figure (if None, figure is not saved)

    Returns:
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Process each site
    for site_id in site_ids:
        site_data = df[df['SCATS_Site'] == site_id].copy()

        if len(site_data) == 0:
            print(f"No data found for site {site_id}")
            continue

        # Get site location for label
        location = site_data['Location'].iloc[0] if 'Location' in site_data.columns else f"Site {site_id}"

        # Extract time component based on grouping
        if group_by == 'hour':
            site_data['group'] = pd.to_datetime(site_data[date_column]).dt.hour
            x_label = 'Hour of Day'
        elif group_by == 'day':
            site_data['group'] = pd.to_datetime(site_data[date_column]).dt.day
            x_label = 'Day of Month'
        elif group_by == 'weekday':
            site_data['group'] = pd.to_datetime(site_data[date_column]).dt.dayofweek
            x_label = 'Day of Week'
        else:
            raise ValueError(f"Unsupported group_by value: {group_by}")

        # Group and aggregate
        grouped = site_data.groupby('group')[flow_column].mean()

        # Plot
        ax.plot(grouped.index, grouped.values, label=f"{location}", marker='o', alpha=0.7)

    # Add labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel('Average Traffic Flow')

    if title:
        plt.title(title)
    else:
        plt.title(f'Average Traffic Flow by {group_by.capitalize()}')

    # Set appropriate x-ticks
    if group_by == 'hour':
        plt.xticks(range(0, 24))
    elif group_by == 'weekday':
        plt.xticks(range(0, 7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    plt.grid(True, alpha=0.3)
    plt.legend(title='SCATS Sites')
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

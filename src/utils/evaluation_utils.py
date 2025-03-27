"""
Utility functions for model evaluation in Boroondara Traffic Flow Prediction System.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt

from src.utils.evaluation import evaluate_regression
from src.utils.visualization import (
    plot_prediction_results,
    plot_training_history,
    plot_site_comparison,
    plot_model_performance_summary
)
from src.utils.model_utils import predict_with_models


def evaluate_site_model(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    site_id: str,
    location: Optional[str],
    results_dir: str,
    scaler=None
) -> Dict[str, Dict[str, float]]:
    """Evaluate models on a specific site.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test targets
        site_id: SCATS site ID
        location: Site location description
        results_dir: Directory to save results
        scaler: Scaler object for inverse transforming predictions

    Returns:
        Dictionary mapping model types to evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"Evaluating Models for SCATS Site {site_id}" +
          (f" - {location}" if location else ""))
    print(f"{'='*50}")

    print(f"Test data shape: {X_test.shape}")

    # Generate predictions
    predictions = predict_with_models(models, X_test)

    # Evaluate predictions
    results = {}
    for model_type, y_pred in predictions.items():
        # Apply inverse scaling if provided
        if scaler is not None:
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_pred_original = y_pred.flatten()
            y_test_original = y_test.flatten()

        # Calculate metrics
        metrics = evaluate_regression(y_test_original, y_pred_original)
        results[model_type] = metrics

        # Print metrics
        print(f"\nResults for {model_type.upper()}:")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")

    # Create visualization
    model_names = [model_type.upper() for model_type in predictions.keys()]
    y_preds = []

    for model_type in predictions.keys():
        if scaler is not None:
            y_pred = scaler.inverse_transform(
                predictions[model_type].reshape(-1, 1)).flatten()
        else:
            y_pred = predictions[model_type].flatten()
        y_preds.append(y_pred)

    if scaler is not None:
        y_test_viz = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_test_viz = y_test.flatten()

    # Create visualization directory
    vis_dir = os.path.join(results_dir, "site_predictions")
    os.makedirs(vis_dir, exist_ok=True)

    # Create plot title
    title = f"Traffic Flow Prediction for Site {site_id}"
    if location:
        title += f" - {location}"

    # Plot predictions (first 96 points = 24 hours)
    num_points = min(96, len(y_test_viz))
    fig = plot_prediction_results(
        y_test_viz[:num_points],
        [pred[:num_points] for pred in y_preds],
        model_names,
        start_date="2006-10-01 00:00",
        freq="15min",
        num_points=num_points,
        title=title
    )

    # Save plot
    plot_path = os.path.join(vis_dir, f"site_{site_id}_predictions.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction visualization saved to {plot_path}")

    return results


def create_model_summary(
    site_results: List[Dict[str, Any]],
    results_dir: str
) -> pd.DataFrame:
    """Create and save a summary of model performance across sites.

    Args:
        site_results: List of site evaluation results
        results_dir: Directory to save results

    Returns:
        DataFrame with model summary
    """
    # Create summary directory
    summary_dir = os.path.join(results_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)

    # Prepare summary data
    summary_data = []

    for site_result in site_results:
        site_id = site_result["site_id"]
        location = site_result.get("location", "Unknown")

        for model_type, metrics in site_result["results"].items():
            summary_data.append({
                "site_id": site_id,
                "location": location,
                "model_type": model_type,
                "mape": metrics["mape"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mae": metrics["mae"]
            })

    if not summary_data:
        print("No evaluation data to summarize")
        return pd.DataFrame()

    # Create detailed summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(summary_dir, "all_sites_metrics.csv"), index=False)

    # Create model summary
    model_summary = []
    for model_type, group in summary_df.groupby("model_type"):
        model_summary.append({
            "model_type": model_type,
            "avg_mape": group["mape"].mean(),
            "min_mape": group["mape"].min(),
            "max_mape": group["mape"].max(),
            "avg_rmse": group["rmse"].mean(),
            "min_rmse": group["rmse"].min(),
            "max_rmse": group["rmse"].max(),
            "avg_r2": group["r2"].mean(),
            "min_r2": group["r2"].min(),
            "max_r2": group["r2"].max()
        })

    model_summary_df = pd.DataFrame(model_summary)
    model_summary_df.to_csv(os.path.join(summary_dir, "model_performance_summary.csv"), index=False)

    # Print model summary
    print("\nOverall Model Performance (Average across all sites):")
    print(model_summary_df[["model_type", "avg_mape", "avg_rmse", "avg_r2"]].to_string(index=False))

    # Generate comparison visualizations
    vis_dir = os.path.join(results_dir, "comparisons")
    os.makedirs(vis_dir, exist_ok=True)

    # Site comparison plot
    site_fig = plot_site_comparison(
        summary_df,
        metric="mape",
        top_n=10,
        save_path=os.path.join(vis_dir, "site_mape_comparison.png")
    )
    plt.close(site_fig)

    # Model performance summary plot
    model_fig = plot_model_performance_summary(
        model_summary_df,
        metrics=["mape", "rmse", "r2"],
        save_path=os.path.join(vis_dir, "model_performance_summary.png")
    )
    plt.close(model_fig)

    print(f"\nComparison visualizations saved to {vis_dir}")

    return model_summary_df


def visualize_training_history(
    history: Dict[str, List[float]],
    model_type: str,
    results_dir: str,
    site_id: Optional[str] = None
) -> None:
    """Visualize and save training history.

    Args:
        history: Training history dictionary
        model_type: Type of model trained
        results_dir: Directory to save results
        site_id: SCATS site ID (if site-specific model)
    """
    # Create history directory
    if site_id:
        history_dir = os.path.join(results_dir, "training_history", "per_site", site_id)
        os.makedirs(history_dir, exist_ok=True)
        output_path = os.path.join(history_dir, f"{model_type}_history.png")
    else:
        history_dir = os.path.join(results_dir, "training_history")
        os.makedirs(history_dir, exist_ok=True)
        output_path = os.path.join(history_dir, f"{model_type}_history.png")

    # Create title
    title = f"{model_type.upper()} Training History"
    if site_id:
        title += f" - Site {site_id}"

    # Plot history
    fig = plot_training_history(history, title=title)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training history visualization saved to {output_path}")

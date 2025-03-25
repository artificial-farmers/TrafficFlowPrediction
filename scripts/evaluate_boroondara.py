"""
Evaluation script for Boroondara traffic flow prediction models.
Compares model performance and visualizes results.
"""
from data.boroondara_preprocessing import load_scats_data, prepare_site_data
from src.utils.visualization import plot_prediction_results
from src.utils.evaluation import evaluate_regression
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Dict, Any, Tuple

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_trained_models(models_dir: str, model_types: List[str]) -> Dict[str, tf.keras.Model]:
    """
    Load trained models from disk.

    Args:
        models_dir: Directory containing trained models
        model_types: List of model types to load

    Returns:
        Dictionary mapping model types to loaded models
    """
    models = {}

    for model_type in model_types:
        model_path = os.path.join(models_dir, f"{model_type}_boroondara.h5")

        if os.path.exists(model_path):
            print(f"Loading {model_type} model from {model_path}")
            models[model_type] = tf.keras.models.load_model(model_path)
        else:
            print(f"Warning: Model file not found at {model_path}")

    return models


def reshape_input_for_rnn(X: np.ndarray) -> np.ndarray:
    """
    Reshape input data for recurrent neural networks (LSTM, GRU).

    Args:
        X: Input features

    Returns:
        Reshaped input
    """
    return X.reshape(X.shape[0], X.shape[1], 1)


def generate_predictions(
    models: Dict[str, tf.keras.Model],
    X_test: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Generate predictions for all models.

    Args:
        models: Dictionary of loaded models
        X_test: Test features

    Returns:
        Dictionary mapping model types to predictions
    """
    predictions = {}

    for model_type, model in models.items():
        print(f"Generating predictions for {model_type}...")

        # Reshape input for RNN models
        if model_type in ['lstm', 'gru']:
            X_test_reshaped = reshape_input_for_rnn(X_test)
        else:
            X_test_reshaped = X_test

        # Generate predictions
        predictions[model_type] = model.predict(X_test_reshaped)

    return predictions


def evaluate_models(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    scaler=None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model predictions.

    Args:
        predictions: Dictionary of model predictions
        y_test: True target values
        scaler: Scaler used to normalize data, if provided will inverse transform

    Returns:
        Dictionary mapping model types to evaluation metrics
    """
    results = {}

    for model_type, y_pred in predictions.items():
        print(f"\nEvaluating {model_type}...")

        # If scaler is provided, inverse transform predictions and true values
        if scaler is not None:
            y_pred_original = scaler.inverse_transform(
                y_pred.reshape(-1, 1)).flatten()
            y_test_original = scaler.inverse_transform(
                y_test.reshape(-1, 1)).flatten()
        else:
            y_pred_original = y_pred.flatten()
            y_test_original = y_test.flatten()

        # Evaluate predictions
        metrics = evaluate_regression(y_test_original, y_pred_original)

        # Print metrics
        print(f"Explained Variance Score: {metrics['explained_variance']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")

        # Store results
        results[model_type] = metrics

    return results


def create_comparison_table(
    evaluation_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Create a comparison table of evaluation metrics for all models.

    Args:
        evaluation_results: Dictionary mapping model types to evaluation metrics

    Returns:
        DataFrame with model comparison
    """
    # Initialize DataFrame
    metrics = ['explained_variance', 'mape', 'mae', 'mse', 'rmse', 'r2']
    metric_names = ['Explained Variance',
                    'MAPE (%)', 'MAE', 'MSE', 'RMSE', 'R²']

    comparison = pd.DataFrame(index=metric_names)

    # Add metrics for each model
    for model_type, results in evaluation_results.items():
        model_metrics = []

        for metric in metrics:
            value = results[metric]

            # Format metric value
            if metric == 'mape':
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value:.4f}"

            model_metrics.append(formatted_value)

        comparison[model_type.upper()] = model_metrics

    return comparison


def visualize_predictions(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    output_dir: str,
    scaler=None,
    num_points: int = 96  # 24 hours of 15-minute intervals
) -> None:
    """
    Visualize model predictions.

    Args:
        predictions: Dictionary of model predictions
        y_test: True target values
        output_dir: Directory to save visualizations
        scaler: Scaler used to normalize data
        num_points: Number of data points to visualize
    """
    os.makedirs(output_dir, exist_ok=True)

    # If scaler is provided, inverse transform predictions and true values
    if scaler is not None:
        y_test_plot = scaler.inverse_transform(
            y_test[:num_points].reshape(-1, 1)).flatten()
        y_preds_plot = []

        for model_type, y_pred in predictions.items():
            y_pred_plot = scaler.inverse_transform(
                y_pred[:num_points].reshape(-1, 1)).flatten()
            y_preds_plot.append(y_pred_plot)
    else:
        y_test_plot = y_test[:num_points].flatten()
        y_preds_plot = [y_pred[:num_points].flatten()
                        for y_pred in predictions.values()]

    # Plot results
    model_names = [name.upper() for name in predictions.keys()]

    fig = plot_prediction_results(
        y_test_plot,
        y_preds_plot,
        model_names,
        start_date="2006-10-01 00:00",
        freq="15min",
        num_points=num_points
    )

    # Save plot
    plot_path = os.path.join(output_dir, "prediction_comparison.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPrediction visualization saved to {plot_path}")


def evaluate_site_specific(
    models: Dict[str, tf.keras.Model],
    scats_data_path: str,
    site_id: str,
    seq_length: int = 12,
    output_dir: str = None
) -> None:
    """
    Evaluate models on a specific SCATS site.

    Args:
        models: Dictionary of loaded models
        scats_data_path: Path to SCATS data CSV file
        site_id: SCATS site ID to evaluate
        seq_length: Number of time steps in sequence
        output_dir: Directory to save visualizations
    """
    print(f"\n{'='*50}")
    print(f"Evaluating models on SCATS site {site_id}")
    print(f"{'='*50}")

    # Load and prepare site data
    scats_df = load_scats_data(scats_data_path)
    X_train, y_train, X_test, y_test, scaler = prepare_site_data(
        scats_df, site_id, seq_length)

    print(f"\nData shapes for site {site_id}:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    # Generate predictions
    predictions = generate_predictions(models, X_test)

    # Evaluate predictions
    evaluation_results = evaluate_models(predictions, y_test, scaler)

    # Create comparison table
    comparison = create_comparison_table(evaluation_results)
    print("\nModel Comparison:")
    print(comparison)

    # Visualize predictions
    if output_dir:
        site_output_dir = os.path.join(output_dir, f"site_{site_id}")
        visualize_predictions(predictions, y_test, site_output_dir, scaler)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Boroondara traffic flow prediction models")

    parser.add_argument(
        "--models_dir",
        type=str,
        default="saved_models/boroondara",
        help="Directory containing trained models"
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed/boroondara",
        help="Directory containing processed data"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/boroondara",
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--model_types",
        type=str,
        default="lstm,gru,saes",
        help="Comma-separated list of model types to evaluate"
    )

    parser.add_argument(
        "--site_specific",
        action="store_true",
        help="Perform site-specific evaluation"
    )

    parser.add_argument(
        "--scats_data",
        type=str,
        default="data/raw/Scats Data October 2006.csv",
        help="Path to SCATS data CSV file (for site-specific evaluation)"
    )

    parser.add_argument(
        "--site_id",
        type=str,
        default="2000",
        help="SCATS site ID for site-specific evaluation"
    )

    return parser.parse_args()


def main():
    """
    Main function to evaluate Boroondara traffic flow prediction models.
    """
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get model types to evaluate
    model_types = [model_type.strip()
                   for model_type in args.model_types.split(',')]

    # Load trained models
    models = load_trained_models(args.models_dir, model_types)

    if not models:
        print("Error: No models could be loaded. Exiting.")
        return

    # If site-specific evaluation is requested
    if args.site_specific:
        evaluate_site_specific(
            models,
            args.scats_data,
            args.site_id,
            output_dir=args.output_dir
        )
        return

    # Otherwise, evaluate on the combined dataset
    # Load test data
    X_test = np.load(os.path.join(args.processed_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.processed_dir, 'y_test.npy'))

    print(f"\nData shapes:")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    # Generate predictions
    predictions = generate_predictions(models, X_test)

    # Evaluate predictions
    evaluation_results = evaluate_models(predictions, y_test)

    # Create comparison table
    comparison = create_comparison_table(evaluation_results)

    print("\nModel Comparison:")
    print(comparison)

    # Save comparison table
    comparison_path = os.path.join(args.output_dir, "model_comparison.csv")
    comparison.to_csv(comparison_path)
    print(f"\nComparison table saved to {comparison_path}")

    # Visualize predictions
    visualize_predictions(predictions, y_test, args.output_dir)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()

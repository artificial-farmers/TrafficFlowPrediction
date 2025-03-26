"""
Evaluation script for Boroondara traffic flow prediction models.
Compares model performance and visualizes results for each SCATS site.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Dict, Any, Tuple, Optional
import joblib

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.boroondara_preprocessing import load_scats_data, prepare_site_data, get_available_scats_sites
from src.utils.visualization import plot_prediction_results
from src.utils.evaluation import evaluate_regression


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
            try:
                # Define custom objects for compatibility with different Keras versions
                custom_objects = {
                    'mse': tf.keras.losses.MeanSquaredError(),
                    'mean_squared_error': tf.keras.losses.MeanSquaredError(),
                    'mape': tf.keras.metrics.MeanAbsolutePercentageError()
                }

                models[model_type] = tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects
                )
            except Exception as e:
                print(f"Error loading {model_type} model: {str(e)}")
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
    site_id: str = None,
    location: str = None,
    scaler=None,
    num_points: int = 96  # 24 hours of 15-minute intervals
) -> None:
    """
    Visualize model predictions.

    Args:
        predictions: Dictionary of model predictions
        y_test: True target values
        output_dir: Directory to save visualizations
        site_id: SCATS site ID
        location: Location description
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
    title = f"Traffic Flow Prediction for Site {site_id}" if site_id else "Traffic Flow Prediction"
    if location:
        title += f" - {location}"

    fig = plot_prediction_results(
        y_test_plot,
        y_preds_plot,
        model_names,
        start_date="2006-10-01 00:00",
        freq="15min",
        num_points=num_points,
        title=title
    )

    # Save plot
    filename = f"site_{site_id}_predictions.png" if site_id else "prediction_comparison.png"
    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPrediction visualization saved to {plot_path}")


def evaluate_site_specific(
    models: Dict[str, tf.keras.Model],
    processed_dir: str,
    output_dir: str,
    site_id: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate models on a specific SCATS site.

    Args:
        models: Dictionary of loaded models
        processed_dir: Directory containing processed data
        output_dir: Directory to save visualizations
        site_id: SCATS site ID to evaluate

    Returns:
        Dictionary with evaluation results
    """
    site_dir = os.path.join(processed_dir, 'per_site', site_id)

    if not os.path.exists(site_dir):
        print(f"Error: No processed data found for site {site_id} at {site_dir}")
        return {}

    # Load site data
    try:
        X_test = np.load(os.path.join(site_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(site_dir, 'y_test.npy'))
        scaler = joblib.load(os.path.join(site_dir, 'scaler.joblib'))

        # Get site info
        site_info_df = pd.read_csv(os.path.join(processed_dir, 'per_site', 'site_info.csv'))
        site_row = site_info_df[site_info_df['site_id'] == site_id]
        location = site_row['location'].iloc[0] if not site_row.empty else "Unknown"

        print(f"\n{'='*50}")
        print(f"Evaluating models on SCATS site {site_id} - {location}")
        print(f"{'='*50}")

        print(f"\nData shapes for site {site_id}:")
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

        # Save comparison table
        os.makedirs(os.path.join(output_dir, 'site_evaluations'), exist_ok=True)
        comparison_path = os.path.join(output_dir, 'site_evaluations', f"site_{site_id}_comparison.csv")
        comparison.to_csv(comparison_path)

        # Visualize predictions
        site_output_dir = os.path.join(output_dir, 'site_visualizations')
        os.makedirs(site_output_dir, exist_ok=True)
        visualize_predictions(predictions, y_test, site_output_dir, site_id, location, scaler)

        return evaluation_results

    except Exception as e:
        print(f"Error evaluating site {site_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def evaluate_all_sites(
    models: Dict[str, tf.keras.Model],
    processed_dir: str,
    output_dir: str,
    max_sites: Optional[int] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate models on all available sites.

    Args:
        models: Dictionary of loaded models
        processed_dir: Directory containing processed data
        output_dir: Directory to save visualizations
        max_sites: Maximum number of sites to evaluate

    Returns:
        Dictionary mapping site IDs to evaluation results
    """
    per_site_dir = os.path.join(processed_dir, 'per_site')

    if not os.path.exists(per_site_dir):
        print(f"Error: Per-site processed data directory not found at {per_site_dir}")
        return {}

    # Get available sites
    try:
        site_info_df = pd.read_csv(os.path.join(per_site_dir, 'site_info.csv'))
        available_sites = site_info_df['site_id'].tolist()
    except:
        # Fallback: just use directory names
        available_sites = [d for d in os.listdir(per_site_dir)
                          if os.path.isdir(os.path.join(per_site_dir, d))]

    if max_sites and max_sites < len(available_sites):
        print(f"Limiting evaluation to {max_sites} out of {len(available_sites)} available sites")
        available_sites = available_sites[:max_sites]

    print(f"\n{'='*50}")
    print(f"Evaluating models on {len(available_sites)} SCATS sites")
    print(f"{'='*50}")

    # Prepare results structure
    all_results = {}

    # Create summary dataframe
    summary_data = []

    # Evaluate each site
    for site_id in available_sites:
        results = evaluate_site_specific(models, processed_dir, output_dir, site_id)
        if results:
            all_results[site_id] = results

            # Add to summary
            for model_type, metrics in results.items():
                summary_data.append({
                    'site_id': site_id,
                    'model_type': model_type,
                    'mape': metrics['mape'],
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2']
                })

    # Create and save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Group by model_type and calculate averages
        model_summaries = []
        for model_type, group in summary_df.groupby('model_type'):
            model_summaries.append({
                'model_type': model_type,
                'avg_mape': group['mape'].mean(),
                'avg_rmse': group['rmse'].mean(),
                'avg_r2': group['r2'].mean(),
                'min_mape': group['mape'].min(),
                'max_mape': group['mape'].max()
            })

        model_summary_df = pd.DataFrame(model_summaries)

        print("\nOverall Model Performance (Average across all sites):")
        print(model_summary_df.to_string(index=False, float_format="%.4f"))

        # Save summary
        os.makedirs(os.path.join(output_dir, 'summaries'), exist_ok=True)
        summary_df.to_csv(os.path.join(output_dir, 'summaries', 'all_sites_metrics.csv'), index=False)
        model_summary_df.to_csv(os.path.join(output_dir, 'summaries', 'model_performance_summary.csv'), index=False)

        print(f"\nEvaluation summaries saved to {os.path.join(output_dir, 'summaries')}")

    return all_results


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Boroondara traffic flow prediction models"
    )

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
        "--site_id",
        type=str,
        default=None,
        help="Specific SCATS site ID to evaluate (if not specified, all sites are evaluated)"
    )

    parser.add_argument(
        "--max_sites",
        type=int,
        default=None,
        help="Maximum number of sites to evaluate"
    )

    parser.add_argument(
        "--scats_data",
        type=str,
        default="data/raw/Scats Data October 2006.csv",
        help="Path to SCATS data CSV file (for site-specific evaluation)"
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

    # If a specific site is requested
    if args.site_id:
        evaluate_site_specific(
            models,
            args.processed_dir,
            args.output_dir,
            args.site_id
        )
    else:
        # Evaluate all sites
        evaluate_all_sites(
            models,
            args.processed_dir,
            args.output_dir,
            args.max_sites
        )

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()

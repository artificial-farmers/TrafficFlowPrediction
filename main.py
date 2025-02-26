"""
Traffic Flow Prediction with Neural Networks (LSTM, GRU, SAE).
Main evaluation script to compare model performance.
"""
import argparse
import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple

from data.preprocessing import process_data
from src.utils.evaluation import evaluate_regression, mean_absolute_percentage_error

warnings.filterwarnings("ignore")


def plot_results(y_true: np.ndarray, y_preds: List[np.ndarray], names: List[str],
                 start_date: str = '2016-3-4 00:00', freq: str = '5min',
                 num_points: int = 288) -> None:
    """Plot the true data and predicted data.

    Args:
        y_true: True values
        y_preds: List of predictions from different models
        names: Model names
        start_date: Starting date for x-axis labels
        freq: Frequency of data points
        num_points: Number of points to plot
    """
    # Create time index
    x = pd.date_range(start_date, periods=num_points, freq=freq)

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Plot ground truth
    ax.plot(x, y_true[:num_points], label="True Data", linewidth=2)

    # Plot predictions
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred[:num_points], label=name, linestyle="--", alpha=0.8)

    # Add legend and grid
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time of Day")
    plt.ylabel("Flow")

    # Format x-axis with hours and minutes
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate traffic flow prediction models"
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default="data/raw/train.csv",
        help="Path to training data CSV file",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default="data/raw/test.csv",
        help="Path to test data CSV file",
    )

    parser.add_argument(
        "--lag",
        type=int,
        default=12,
        help="Number of lagged observations to use as features",
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        default="saved_models",
        help="Directory containing trained models",
    )

    return parser.parse_args()


def main():
    """Main function to evaluate models."""
    # Parse arguments
    args = parse_arguments()

    # Process data
    _, _, X_test, y_test, scaler = process_data(
        args.train_file, args.test_file, args.lag
    )

    # Original scale y_test
    y_test_original = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).reshape(1, -1)[0]

    # Load models
    model_names = ["lstm", "gru", "saes"]
    models = []

    for name in model_names:
        model_path = os.path.join(args.models_dir, f"{name}.h5")
        if os.path.exists(model_path):
            models.append(tf.keras.models.load_model(model_path))
        else:
            print(f"Model {name} not found at {model_path}")
            model_names.remove(name)

    # Make predictions
    predictions = []

    for i, (name, model) in enumerate(zip(model_names, models)):
        print(f"Evaluating {name.upper()} model...")

        # Reshape input for recurrent models
        if name in ["lstm", "gru"]:
            X_test_reshaped = np.reshape(
                X_test, (X_test.shape[0], X_test.shape[1], 1)
            )
        else:
            X_test_reshaped = X_test

        # Generate predictions
        y_pred = model.predict(X_test_reshaped)

        # Inverse transform predictions to original scale
        y_pred_original = scaler.inverse_transform(
            y_pred.reshape(-1, 1)
        ).reshape(1, -1)[0]

        predictions.append(y_pred_original)

        # Create model visualization image
        os.makedirs("images", exist_ok=True)
        model_plot_path = f"images/{name}.png"
        tf.keras.utils.plot_model(model, to_file=model_plot_path, show_shapes=True)

        # Evaluate model
        results = evaluate_regression(y_test_original, y_pred_original)

        # Print results
        print(f"\nEvaluation results for {name.upper()}:")
        print(f"Explained Variance Score: {results['explained_variance']:.4f}")
        print(f"MAPE: {results['mape']:.2f}%")
        print(f"MAE: {results['mae']:.4f}")
        print(f"MSE: {results['mse']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"R²: {results['r2']:.4f}")

    # Plot comparison of predictions
    plot_results(y_test_original, predictions, [name.upper() for name in model_names])


if __name__ == "__main__":
    main()

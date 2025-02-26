"""
Command-line script for training traffic flow prediction models.
"""
import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

# Add the root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import process_data
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.saes import SAEModel


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train traffic flow prediction models")

    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "gru", "saes"],
        help="Model to train: lstm, gru, or saes",
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
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models",
        help="Directory to save trained models",
    )

    return parser.parse_args()


def main():
    """Main function to train the model."""
    # Parse arguments
    args = parse_arguments()

    # Process data
    X_train, y_train, _, _, _ = process_data(
        args.train_file, args.test_file, args.lag
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure model
    config = {
        "input_dim": args.lag,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }

    # Train the selected model
    if args.model == "lstm":
        # Reshape input for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Create and train model
        config["lstm_units"] = [64, 64]
        model = LSTMModel(config)
        model.build()
        model.compile()

    elif args.model == "gru":
        # Reshape input for GRU
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Create and train model
        config["gru_units"] = [64, 64]
        model = GRUModel(config)
        model.build()
        model.compile()

    elif args.model == "saes":
        # Create and train model
        config["hidden_dims"] = [400, 400, 400]
        model = SAEModel(config)
        model.build()
        model.pretrain(X_train, y_train)
        model.compile()

    # Train the model
    history = model.train(X_train, y_train)

    # Save model
    model_path = os.path.join(args.output_dir, f"{args.model}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(args.output_dir, f"{args.model}_loss.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")


if __name__ == "__main__":
    main()

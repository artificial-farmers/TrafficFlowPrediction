"""
Main entry point for Boroondara Traffic Flow Prediction System.
Provides a unified interface for data preprocessing, model training,
and evaluation for the Boroondara implementation.
"""
import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime

# Data processing
from data.boroondara_preprocessing import prepare_boroondara_dataset, load_scats_data, prepare_site_data

# Configuration
from config.boroondara_config import get_all_configs

# Models
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.saes import SAEModel

# Evaluation
from src.utils.evaluation import evaluate_regression
from src.utils.visualization import plot_prediction_results, plot_training_history


def setup_environment():
    """Set up the environment for TensorFlow."""
    # Set TensorFlow log level
    tf.get_logger().setLevel('ERROR')

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create necessary directories
    os.makedirs("data/processed/boroondara", exist_ok=True)
    os.makedirs("saved_models/boroondara", exist_ok=True)
    os.makedirs("logs/boroondara", exist_ok=True)
    os.makedirs("results/boroondara", exist_ok=True)

    # Register custom functions for backward compatibility
    # This helps when loading models saved with different versions of TensorFlow/Keras
    try:
        # Define MSE function for compatibility
        @tf.keras.utils.register_keras_serializable(package='keras.metrics')
        def mse(y_true, y_pred):
            return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        # Make it available globally
        tf.keras.metrics.mse = mse

        # Also register the standard name
        @tf.keras.utils.register_keras_serializable(package='keras.metrics')
        def mean_squared_error(y_true, y_pred):
            return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        tf.keras.metrics.mean_squared_error = mean_squared_error
    except Exception as e:
        print(f"Warning: Could not register custom metrics: {str(e)}")
        print("This may cause issues when loading saved models.")


def preprocess_data(args):
    """Preprocess the Boroondara data."""
    print("\n" + "="*50)
    print("Preprocessing Boroondara Data")
    print("="*50)

    prepare_boroondara_dataset(
        args.scats_data,
        args.metadata,
        args.processed_dir,
        args.seq_length
    )
    print("\nData preprocessing completed.")


def train_models(args):
    """Train models on the Boroondara data."""
    print("\n" + "="*50)
    print("Training Models on Boroondara Data")
    print("="*50)

    # Check if processed data exists
    x_train_path = os.path.join(args.processed_dir, 'X_train.npy')
    if not os.path.exists(x_train_path):
        print("Processed data not found. Running preprocessing first...")
        preprocess_data(args)

    # Load data
    X_train = np.load(os.path.join(args.processed_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.processed_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(args.processed_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.processed_dir, 'y_test.npy'))

    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    # Get configurations
    configs = get_all_configs()

    # Determine which models to train
    if args.models == 'all':
        models_to_train = ['lstm', 'gru', 'saes']
    else:
        models_to_train = [model.strip() for model in args.models.split(',')]

    results = []

    # Train each requested model
    for model_type in models_to_train:
        if model_type not in configs:
            print(f"Warning: Unknown model type '{model_type}'. Skipping.")
            continue

        print(f"\n{'='*40}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*40}")

        config = configs[model_type]

        # Add global training parameters
        config['batch_size'] = args.batch_size
        config['epochs'] = args.epochs

        # Create model
        if model_type == 'lstm':
            model = LSTMModel(config)
        elif model_type == 'gru':
            model = GRUModel(config)
        elif model_type == 'saes':
            model = SAEModel(config)

        # Build model
        model.build()

        # For SAE, perform pretraining
        if model_type == 'saes':
            print("\nPre-training SAE layers...")
            model.pretrain(
                X_train,
                y_train,
                validation_split=0.1,
                epochs=args.pretraining_epochs
            )

        # Compile model
        model.compile()

        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=args.patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    args.log_dir,
                    model_type,
                    datetime.now().strftime("%Y%m%d-%H%M%S")
                ),
                histogram_freq=1
            )
        ]

        # Reshape input for RNN models
        if model_type in ['lstm', 'gru']:
            X_train_model = X_train.reshape(
                X_train.shape[0], X_train.shape[1], 1)
            X_test_model = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        else:
            X_train_model = X_train
            X_test_model = X_test

        # Train model
        history = model.train(
            X_train_model,
            y_train,
            validation_split=0.1,
            callbacks=callbacks
        )

        # Save model
        model_path = os.path.join(
            args.model_dir, f"{model_type}_boroondara.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # Evaluate model
        loss, mape = model.model.evaluate(X_test_model, y_test, verbose=1)

        # Store results
        results.append({
            'model_type': model_type,
            'loss': loss,
            'mape': mape,
            'model_path': model_path,
            'history': history.history
        })

        # Plot and save training history
        history_fig = plot_training_history(history.history)
        history_path = os.path.join(
            args.results_dir, f"{model_type}_training_history.png")
        history_fig.savefig(history_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {history_path}")

    # Print summary
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)

    for result in results:
        print(f"\n{result['model_type'].upper()}:")
        print(f"  Test Loss: {result['loss']:.4f}")
        print(f"  Test MAPE: {result['mape']:.4f}")
        print(f"  Model saved: {result['model_path']}")

    print("\nTraining completed successfully!")


def evaluate_models(args):
    """Evaluate models on the Boroondara data."""
    print("\n" + "="*50)
    print("Evaluating Models on Boroondara Data")
    print("="*50)

    # Determine models to evaluate
    if args.models == 'all':
        model_types = ['lstm', 'gru', 'saes']
    else:
        model_types = [model.strip() for model in args.models.split(',')]

    # Load models
    models = {}
    for model_type in model_types:
        model_path = os.path.join(
            args.model_dir, f"{model_type}_boroondara.h5")
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
                print(f"Successfully loaded {model_type} model")
            except Exception as e:
                print(f"Error loading {model_type} model: {str(e)}")
                print(
                    "Try retraining the model with: python boroondara_main.py train --models " + model_type)
        else:
            print(f"Warning: Model not found at {model_path}")

    if not models:
        print("No models found. Train models first using the 'train' command.")
        return

    # If site-specific evaluation is requested
    if args.site_id:
        print(f"\nPerforming site-specific evaluation for site {args.site_id}")

        # Load SCATS data
        scats_df = load_scats_data(args.scats_data)

        try:
            # Prepare site data
            X_train, y_train, X_test, y_test, scaler = prepare_site_data(
                scats_df,
                args.site_id,
                args.seq_length
            )

            # Generate predictions and evaluate
            predictions = {}
            for model_type, model in models.items():
                # Reshape input for RNN models
                if model_type in ['lstm', 'gru']:
                    X_test_model = X_test.reshape(
                        X_test.shape[0], X_test.shape[1], 1)
                else:
                    X_test_model = X_test

                # Generate predictions
                y_pred = model.predict(X_test_model)

                # Inverse transform
                y_pred_original = scaler.inverse_transform(
                    y_pred.reshape(-1, 1)).flatten()
                predictions[model_type] = y_pred_original

            # Inverse transform true values
            y_test_original = scaler.inverse_transform(
                y_test.reshape(-1, 1)).flatten()

            # Evaluate each model
            for model_type, y_pred in predictions.items():
                print(f"\nEvaluation for {model_type.upper()}:")
                metrics = evaluate_regression(y_test_original, y_pred)

                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")

            # Visualize predictions
            model_names = [name.upper() for name in predictions.keys()]
            y_preds = [predictions[model] for model in models.keys()]

            # Plot first 96 points (24 hours)
            fig = plot_prediction_results(
                y_test_original[:96],
                [pred[:96] for pred in y_preds],
                model_names,
                start_date="2006-10-01 00:00",
                freq="15min",
                num_points=96
            )

            # Save plot
            plot_path = os.path.join(
                args.results_dir, f"site_{args.site_id}_predictions.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"\nPrediction visualization saved to {plot_path}")

        except ValueError as e:
            print(f"Error: {e}")

    # Otherwise, evaluate on the combined dataset
    else:
        # Load test data
        X_test = np.load(os.path.join(args.processed_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(args.processed_dir, 'y_test.npy'))

        print(f"\nData shapes:")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")

        # Generate predictions
        predictions = {}
        for model_type, model in models.items():
            # Reshape input for RNN models
            if model_type in ['lstm', 'gru']:
                X_test_model = X_test.reshape(
                    X_test.shape[0], X_test.shape[1], 1)
            else:
                X_test_model = X_test

            # Generate predictions
            predictions[model_type] = model.predict(X_test_model)

        # Evaluate each model
        results = {}
        for model_type, y_pred in predictions.items():
            print(f"\nEvaluation for {model_type.upper()}:")
            metrics = evaluate_regression(y_test, y_pred.flatten())
            results[model_type] = metrics

            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        # Visualize predictions
        model_names = [name.upper() for name in predictions.keys()]
        y_preds = [predictions[model].flatten() for model in models.keys()]

        # Plot first 96 points (24 hours)
        fig = plot_prediction_results(
            y_test[:96],
            [pred[:96] for pred in y_preds],
            model_names,
            start_date="2006-10-01 00:00",
            freq="15min",
            num_points=96
        )

        # Save plot
        plot_path = os.path.join(args.results_dir, "combined_predictions.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"\nPrediction visualization saved to {plot_path}")

    print("\nEvaluation completed successfully!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Boroondara Traffic Flow Prediction System"
    )

    # Main command
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute"
    )

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--scats_data",
        type=str,
        default="data/raw/Scats Data October 2006.csv",
        help="Path to SCATS data CSV file"
    )
    common_parser.add_argument(
        "--metadata",
        type=str,
        default="data/raw/SCATSSiteListingSpreadsheet_VicRoads.csv",
        help="Path to SCATS metadata CSV file"
    )
    common_parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed/boroondara",
        help="Directory for processed data"
    )
    common_parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models/boroondara",
        help="Directory for saved models"
    )
    common_parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/boroondara",
        help="Directory for TensorBoard logs"
    )
    common_parser.add_argument(
        "--results_dir",
        type=str,
        default="results/boroondara",
        help="Directory for results"
    )
    common_parser.add_argument(
        "--seq_length",
        type=int,
        default=12,
        help="Sequence length for prediction"
    )
    common_parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to use (lstm, gru, saes, or all)"
    )

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess data",
        parents=[common_parser]
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train models",
        parents=[common_parser]
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--pretraining_epochs",
        type=int,
        default=50,
        help="Number of pretraining epochs for SAE"
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate models",
        parents=[common_parser]
    )
    evaluate_parser.add_argument(
        "--site_id",
        type=str,
        default=None,
        help="SCATS site ID for site-specific evaluation"
    )

    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Set up the environment
    setup_environment()

    # Execute the requested command
    if args.command == "preprocess":
        preprocess_data(args)
    elif args.command == "train":
        train_models(args)
    elif args.command == "evaluate":
        evaluate_models(args)
    else:
        print("Please specify a command: preprocess, train, or evaluate")
        print("Example: python boroondara_main.py train")


if __name__ == "__main__":
    main()

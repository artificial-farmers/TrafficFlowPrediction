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
import pandas as pd
import joblib
from datetime import datetime

# Data processing
from data.boroondara_preprocessing import (
    prepare_boroondara_dataset, load_scats_data,
    prepare_site_data, get_available_scats_sites
)

# Configuration
from config.boroondara_config import get_all_configs

# Models
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.saes import SAEModel

# Evaluation
from src.utils.evaluation import evaluate_regression
from src.utils.visualization import (
    plot_prediction_results, plot_training_history,
    plot_site_comparison, plot_model_performance_summary,
    plot_traffic_patterns
)


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
        args.seq_length,
        per_site=True
    )
    print("\nData preprocessing completed.")


def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    args,
    site_id: str = None
):
    """Train a single model.

    Args:
        model_type: Type of model to train
        X_train: Training features
        y_train: Training targets
        X_test: Testing features
        y_test: Testing targets
        config: Model configuration
        args: Command line arguments
        site_id: SCATS site ID (if training site-specific model)

    Returns:
        Trained model and training results
    """
    # Ensure site_id is a string if provided
    if site_id is not None:
        site_id = str(site_id)

    print(f"\n{'='*40}")
    if site_id:
        print(f"Training {model_type.upper()} model for site {site_id}")
    else:
        print(f"Training {model_type.upper()} model")
    print(f"{'='*40}")

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
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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
    log_dir = os.path.join(
        args.log_dir,
        model_type,
        f"site_{site_id}" if site_id else "combined",
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
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

    # Evaluate model
    loss, mape = model.model.evaluate(X_test_model, y_test, verbose=1)

    # Save model
    if site_id:
        model_dir = os.path.join(args.model_dir, "per_site", site_id)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_type}.h5")
    else:
        model_path = os.path.join(args.model_dir, f"{model_type}_boroondara.h5")

    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot and save training history
    history_fig = plot_training_history(history.history)

    if site_id:
        history_dir = os.path.join(args.results_dir, "training_history", "per_site", site_id)
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, f"{model_type}.png")
    else:
        history_dir = os.path.join(args.results_dir, "training_history")
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, f"{model_type}.png")

    history_fig.savefig(history_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to {history_path}")

    return model, {
        'model_type': model_type,
        'loss': loss,
        'mape': mape,
        'model_path': model_path,
        'history': history.history
    }


def train_site_specific_models(args, site_id):
    """Train models for a specific SCATS site.

    Args:
        args: Command line arguments
        site_id: SCATS site ID
    """
    # Ensure site_id is a string
    site_id = str(site_id)

    print(f"\n{'='*50}")
    print(f"Training Models for SCATS Site {site_id}")
    print(f"{'='*50}")

    # Load site data
    site_dir = os.path.join(args.processed_dir, "per_site", site_id)

    if not os.path.exists(site_dir):
        print(f"Error: No processed data found for site {site_id}")
        print(f"Run preprocessing first: python boroondara_main.py preprocess")
        return

    try:
        X_train = np.load(os.path.join(site_dir, "X_train.npy"))
        y_train = np.load(os.path.join(site_dir, "y_train.npy"))
        X_test = np.load(os.path.join(site_dir, "X_test.npy"))
        y_test = np.load(os.path.join(site_dir, "y_test.npy"))

        # Get site information
        site_info_path = os.path.join(args.processed_dir, "per_site", "site_info.csv")
        if os.path.exists(site_info_path):
            site_info = pd.read_csv(site_info_path)
            site_row = site_info[site_info["site_id"] == site_id]
            if not site_row.empty:
                location = site_row["location"].iloc[0]
                print(f"Training models for {location}")

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

        # Train each model
        for model_type in models_to_train:
            if model_type not in configs:
                print(f"Warning: Unknown model type '{model_type}'. Skipping.")
                continue

            _, result = train_model(
                model_type,
                X_train,
                y_train,
                X_test,
                y_test,
                configs[model_type],
                args,
                site_id
            )

            results.append(result)

        # Print summary
        print("\n" + "="*50)
        print(f"Training Summary for Site {site_id}")
        print("="*50)

        for result in results:
            print(f"\n{result['model_type'].upper()}:")
            print(f"  Test Loss: {result['loss']:.4f}")
            print(f"  Test MAPE: {result['mape']:.4f}")
            print(f"  Model saved: {result['model_path']}")

    except Exception as e:
        print(f"Error training models for site {site_id}: {str(e)}")
        import traceback
        traceback.print_exc()


def train_models(args):
    """Train models on the Boroondara data."""
    print("\n" + "="*50)
    print("Training Models on Boroondara Data")
    print("="*50)

    # If a specific site is requested
    if args.site_id:
        # Ensure site_id is a string
        args.site_id = str(args.site_id)
        train_site_specific_models(args, args.site_id)
        return

    # Check if processed data exists
    per_site_dir = os.path.join(args.processed_dir, "per_site")
    combined_data_path = os.path.join(args.processed_dir, "X_train.npy")

    if not os.path.exists(per_site_dir) and not os.path.exists(combined_data_path):
        print("Processed data not found. Running preprocessing first...")
        preprocess_data(args)

    # If per_site data exists but no combined data
    if args.combined_model and not os.path.exists(combined_data_path):
        print("Combined data not found. Creating combined dataset...")
        prepare_boroondara_dataset(
            args.scats_data,
            args.metadata,
            args.processed_dir,
            args.seq_length,
            per_site=False
        )

    # Train combined model if requested
    if args.combined_model:
        # Load data
        X_train = np.load(os.path.join(args.processed_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(args.processed_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(args.processed_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(args.processed_dir, 'y_test.npy'))

        print(f"Combined data shapes:")
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

            _, result = train_model(
                model_type,
                X_train,
                y_train,
                X_test,
                y_test,
                configs[model_type],
                args
            )

            results.append(result)

        # Print summary
        print("\n" + "="*50)
        print("Training Summary for Combined Model")
        print("="*50)

        for result in results:
            print(f"\n{result['model_type'].upper()}:")
            print(f"  Test Loss: {result['loss']:.4f}")
            print(f"  Test MAPE: {result['mape']:.4f}")
            print(f"  Model saved: {result['model_path']}")

    # Train per-site models if requested
    if args.train_all_sites:
        # Get list of available sites
        try:
            site_info_path = os.path.join(args.processed_dir, "per_site", "site_info.csv")
            if os.path.exists(site_info_path):
                site_info = pd.read_csv(site_info_path)
                sites = site_info["site_id"].tolist()
            else:
                # Fallback: Get directories from processed/per_site
                sites = [d for d in os.listdir(per_site_dir) if os.path.isdir(os.path.join(per_site_dir, d))]

            print(f"Found {len(sites)} sites with processed data")

            # Limit number of sites if specified
            if args.max_sites and args.max_sites < len(sites):
                print(f"Limiting to {args.max_sites} sites as specified")
                sites = sites[:args.max_sites]

            # Train models for each site
            for site_id in sites:
                # Ensure site_id is a string
                site_id = str(site_id)
                train_site_specific_models(args, site_id)

        except Exception as e:
            print(f"Error training per-site models: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nTraining completed successfully!")


def load_models(model_dir, model_types, site_id=None):
    """Load trained models.

    Args:
        model_dir: Directory containing trained models
        model_types: List of model types to load
        site_id: SCATS site ID (if loading site-specific models)

    Returns:
        Dictionary of loaded models
    """
    models = {}

    # Ensure site_id is a string if provided
    if site_id is not None:
        site_id = str(site_id)

    # Determine the model directory based on whether it's site-specific
    if site_id:
        load_dir = os.path.join(model_dir, "per_site", site_id)
    else:
        load_dir = model_dir

    if not os.path.exists(load_dir):
        print(f"Warning: Model directory {load_dir} does not exist")
        return models

    for model_type in model_types:
        if site_id:
            model_path = os.path.join(load_dir, f"{model_type}.h5")
        else:
            model_path = os.path.join(load_dir, f"{model_type}_boroondara.h5")

        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} does not exist")
            continue

        try:
            # Define custom objects for compatibility with different Keras versions
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mean_squared_error': tf.keras.losses.MeanSquaredError(),
                'mape': tf.keras.metrics.MeanAbsolutePercentageError()
            }

            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects
            )

            models[model_type] = model
            print(f"Successfully loaded {model_type} model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")

    return models


def evaluate_site(models, site_id, args):
    """Evaluate models on a specific site.

    Args:
        models: Dictionary of trained models
        site_id: SCATS site ID
        args: Command line arguments

    Returns:
        Evaluation results
    """
    # Ensure site_id is a string
    site_id = str(site_id)

    print(f"\n{'='*50}")
    print(f"Evaluating Models for SCATS Site {site_id}")
    print(f"{'='*50}")

    # Load site data
    site_dir = os.path.join(args.processed_dir, "per_site", site_id)

    if not os.path.exists(site_dir):
        print(f"Error: No processed data found for site {site_id}")
        return None

    try:
        X_test = np.load(os.path.join(site_dir, "X_test.npy"))
        y_test = np.load(os.path.join(site_dir, "y_test.npy"))
        scaler = joblib.load(os.path.join(site_dir, "scaler.joblib"))

        # Get site information
        site_info_path = os.path.join(args.processed_dir, "per_site", "site_info.csv")
        location = None
        if os.path.exists(site_info_path):
            site_info = pd.read_csv(site_info_path)
            site_row = site_info[site_info["site_id"] == site_id]
            if not site_row.empty:
                location = site_row["location"].iloc[0]
                print(f"Evaluating models for {location}")

        print(f"Test data shape: {X_test.shape}")

        # Get predictions from each model
        predictions = {}

        for model_type, model in models.items():
            # Reshape input for RNN models
            if model_type in ['lstm', 'gru']:
                X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            else:
                X_test_reshaped = X_test

            # Generate predictions
            y_pred = model.predict(X_test_reshaped)
            predictions[model_type] = y_pred

        # Evaluate predictions
        results = {}
        for model_type, y_pred in predictions.items():
            # Apply inverse scaling
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

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
            y_pred = predictions[model_type]
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_preds.append(y_pred_original)

        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Visualization directory
        vis_dir = os.path.join(args.results_dir, "site_predictions")
        os.makedirs(vis_dir, exist_ok=True)

        # Create plot title
        title = f"Traffic Flow Prediction for Site {site_id}"
        if location:
            title += f" - {location}"

        # Plot predictions (first 96 points = 24 hours)
        fig = plot_prediction_results(
            y_test_original[:96],
            [pred[:96] for pred in y_preds],
            model_names,
            start_date="2006-10-01 00:00",
            freq="15min",
            num_points=96,
            title=title
        )

        # Save plot
        plot_path = os.path.join(vis_dir, f"site_{site_id}_predictions.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Prediction visualization saved to {plot_path}")

        return {
            "site_id": site_id,
            "location": location,
            "results": results
        }

    except Exception as e:
        print(f"Error evaluating site {site_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_models(args):
    """Evaluate models on the Boroondara data."""
    print("\n" + "="*50)
    print("Evaluating Models on Boroondara Data")
    print("="*50)

    # Determine which models to evaluate
    if args.models == 'all':
        model_types = ['lstm', 'gru', 'saes']
    else:
        model_types = [model.strip() for model in args.models.split(',')]

    # Evaluate for a specific site
    if args.site_id:
        # Ensure site_id is a string
        args.site_id = str(args.site_id)

        # Load site-specific models if they exist
        site_model_dir = os.path.join(args.model_dir, "per_site", args.site_id)
        if os.path.exists(site_model_dir):
            models = load_models(args.model_dir, model_types, args.site_id)

            if not models:
                print(f"No site-specific models found for site {args.site_id}")
                print("Falling back to combined models")
                models = load_models(args.model_dir, model_types)
        else:
            # Fallback to combined models
            print(f"No site-specific models found for site {args.site_id}")
            print("Using combined models")
            models = load_models(args.model_dir, model_types)

        if not models:
            print("No models could be loaded. Please train models first.")
            return

        evaluate_site(models, args.site_id, args)

    # Evaluate all sites
    elif args.evaluate_all_sites:
        # Load combined models
        models = load_models(args.model_dir, model_types)

        if not models:
            print("No models could be loaded. Please train models first.")
            return

        # Get all available sites
        try:
            site_info_path = os.path.join(args.processed_dir, "per_site", "site_info.csv")
            if os.path.exists(site_info_path):
                site_info = pd.read_csv(site_info_path)
                sites = site_info["site_id"].tolist()
            else:
                # Fallback: Get directories from processed/per_site
                per_site_dir = os.path.join(args.processed_dir, "per_site")
                sites = [d for d in os.listdir(per_site_dir)
                         if os.path.isdir(os.path.join(per_site_dir, d))]

            print(f"Found {len(sites)} sites with processed data")

            # Limit number of sites if specified
            if args.max_sites and args.max_sites < len(sites):
                print(f"Limiting to {args.max_sites} sites as specified")
                sites = sites[:args.max_sites]

            # Evaluate each site
            all_results = []
            summary_data = []

            for site_id in sites:
                # Ensure site_id is a string
                site_id = str(site_id)
                site_result = evaluate_site(models, site_id, args)
                if site_result:
                    all_results.append(site_result)

                    # Add to summary data
                    for model_type, metrics in site_result["results"].items():
                        summary_data.append({
                            "site_id": site_id,
                            "location": site_result["location"],
                            "model_type": model_type,
                            "mape": metrics["mape"],
                            "rmse": metrics["rmse"],
                            "r2": metrics["r2"],
                            "mae": metrics["mae"]
                        })

            # Create and save summary
            if summary_data:
                # Create summary directory
                summary_dir = os.path.join(args.results_dir, "summaries")
                os.makedirs(summary_dir, exist_ok=True)

                # Save detailed summary
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
                vis_dir = os.path.join(args.results_dir, "comparisons")
                os.makedirs(vis_dir, exist_ok=True)

                # Site comparison plot
                site_fig = plot_site_comparison(
                    summary_df,
                    metric="mape",
                    top_n=10,
                    save_path=os.path.join(vis_dir, "site_mape_comparison.png")
                )

                # Model performance summary plot
                model_fig = plot_model_performance_summary(
                    model_summary_df,
                    metrics=["mape", "rmse", "r2"],
                    save_path=os.path.join(vis_dir, "model_performance_summary.png")
                )

                print(f"\nComparison visualizations saved to {vis_dir}")

        except Exception as e:
            print(f"Error evaluating sites: {str(e)}")
            import traceback
            traceback.print_exc()

    # Evaluate combined model
    else:
        # Load combined models
        models = load_models(args.model_dir, model_types)

        if not models:
            print("No models could be loaded. Please train models first.")
            return

        # Load combined test data
        try:
            X_test = np.load(os.path.join(args.processed_dir, "X_test.npy"))
            y_test = np.load(os.path.join(args.processed_dir, "y_test.npy"))

            print(f"Test data shape: {X_test.shape}")

            # Get predictions from each model
            predictions = {}

            for model_type, model in models.items():
                # Reshape input for RNN models
                if model_type in ['lstm', 'gru']:
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                else:
                    X_test_reshaped = X_test

                # Generate predictions
                y_pred = model.predict(X_test_reshaped)
                predictions[model_type] = y_pred

            # Evaluate predictions
            results = {}
            for model_type, y_pred in predictions.items():
                # Calculate metrics (data is already scaled)
                metrics = evaluate_regression(y_test, y_pred.flatten())
                results[model_type] = metrics

                # Print metrics
                print(f"\nResults for {model_type.upper()}:")
                print(f"  MAPE: {metrics['mape']:.2f}%")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  R²: {metrics['r2']:.4f}")

            # Create visualization
            model_names = [model_type.upper() for model_type in predictions.keys()]
            y_preds = [y_pred.flatten() for y_pred in predictions.values()]

            # Visualization directory
            vis_dir = os.path.join(args.results_dir, "combined")
            os.makedirs(vis_dir, exist_ok=True)

            # Plot predictions (first 96 points = 24 hours)
            fig = plot_prediction_results(
                y_test[:96],
                [pred[:96] for pred in y_preds],
                model_names,
                start_date="2006-10-01 00:00",
                freq="15min",
                num_points=96,
                title="Combined Traffic Flow Prediction"
            )

            # Save plot
            plot_path = os.path.join(vis_dir, "combined_predictions.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Prediction visualization saved to {plot_path}")

        except Exception as e:
            print(f"Error evaluating combined model: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nEvaluation completed successfully!")


def visualize_data(args):
    """Create visualizations of the raw traffic data."""
    print("\n" + "="*50)
    print("Visualizing Traffic Flow Data")
    print("="*50)

    # Create visualization directory
    vis_dir = os.path.join(args.results_dir, "data_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    try:
        # Load SCATS data
        scats_df = load_scats_data(args.scats_data)
        print(f"Loaded {len(scats_df)} records from SCATS data")

        # Get available sites
        available_sites = get_available_scats_sites(scats_df)
        print(f"Found {len(available_sites)} unique SCATS sites in the dataset")

        # Determine which sites to visualize
        sites_to_visualize = []

        if args.site_id:
            # Ensure site_id is a string
            args.site_id = str(args.site_id)

            if args.site_id in available_sites:
                sites_to_visualize = [args.site_id]
            else:
                print(f"Warning: Site {args.site_id} not found in the dataset")
                return
        else:
            # Get top sites by data volume
            site_counts = scats_df['SCATS_Site'].value_counts().head(args.max_sites or 5)
            sites_to_visualize = site_counts.index.tolist()

        print(f"Visualizing traffic patterns for {len(sites_to_visualize)} sites")

        # Create hourly pattern visualization
        hourly_fig = plot_traffic_patterns(
            scats_df,
            sites_to_visualize,
            group_by='hour',
            title='Hourly Traffic Flow Patterns',
            save_path=os.path.join(vis_dir, "hourly_patterns.png")
        )

        # Create daily pattern visualization
        daily_fig = plot_traffic_patterns(
            scats_df,
            sites_to_visualize,
            group_by='weekday',
            title='Daily Traffic Flow Patterns',
            save_path=os.path.join(vis_dir, "daily_patterns.png")
        )

        print(f"Traffic pattern visualizations saved to {vis_dir}")

    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()


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
    common_parser.add_argument(
        "--site_id",
        type=str,
        default=None,
        help="Specific SCATS site ID"
    )
    common_parser.add_argument(
        "--max_sites",
        type=int,
        default=None,
        help="Maximum number of sites to process"
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
    train_parser.add_argument(
        "--combined_model",
        action="store_true",
        help="Train a combined model using data from all sites"
    )
    train_parser.add_argument(
        "--train_all_sites",
        action="store_true",
        help="Train separate models for each site"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate models",
        parents=[common_parser]
    )
    evaluate_parser.add_argument(
        "--evaluate_all_sites",
        action="store_true",
        help="Evaluate on all available sites"
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Create data visualizations",
        parents=[common_parser]
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
    elif args.command == "visualize":
        visualize_data(args)
    else:
        print("Please specify a command: preprocess, train, evaluate, or visualize")
        print("Example: python boroondara_main.py train --site_id 0970")


if __name__ == "__main__":
    main()

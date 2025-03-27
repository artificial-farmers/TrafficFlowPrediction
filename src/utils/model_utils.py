"""
Utility functions for model management in Boroondara Traffic Flow Prediction System.
"""
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.saes import SAEModel
from src.models.transformer import TransformerModel, PositionalEncoding
from config.boroondara_config import get_all_configs
from src.utils.data_processor import reshape_for_rnn


def setup_model_directories(model_dir: str, log_dir: str, results_dir: str) -> None:
    """Create necessary model directories if they don't exist.

    Args:
        model_dir: Directory for saved models
        log_dir: Directory for TensorBoard logs
        results_dir: Directory for results
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "per_site"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "training_history"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "site_predictions"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "summaries"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "comparisons"), exist_ok=True)


def setup_tensorflow() -> None:
    """Set up TensorFlow environment.
    """
    # Set TensorFlow log level
    tf.get_logger().setLevel('ERROR')

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Register custom metrics for compatibility
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


def create_model(model_type: str, config: Dict[str, Any]) -> Any:
    """Create a model instance based on the specified type.

    Args:
        model_type: Type of model to create
        config: Model configuration dictionary

    Returns:
        Model instance
    """
    if model_type == 'lstm':
        return LSTMModel(config)
    elif model_type == 'gru':
        return GRUModel(config)
    elif model_type == 'saes':
        return SAEModel(config)
    elif model_type == 'transformer':
        return TransformerModel(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_callbacks(
    model_type: str,
    patience: int,
    log_dir: str,
    site_id: Optional[str] = None
) -> List[tf.keras.callbacks.Callback]:
    """Create training callbacks.

    Args:
        model_type: Type of model being trained
        patience: Patience for early stopping
        log_dir: Directory for TensorBoard logs
        site_id: SCATS site ID (if training site-specific model)

    Returns:
        List of callbacks
    """
    # Ensure site_id is a string if provided
    if site_id is not None:
        site_id = str(site_id)

    # Create callback directory path
    callback_dir = os.path.join(
        log_dir,
        model_type,
        f"site_{site_id}" if site_id else "combined",
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # Create callbacks
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=callback_dir,
            histogram_freq=1
        ),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(callback_dir, 'checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    return callbacks


def load_models(
    model_dir: str,
    model_types: List[str],
    site_id: Optional[str] = None
) -> Dict[str, tf.keras.Model]:
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
            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} does not exist")
                continue
        else:
            # First try standard naming pattern
            model_path = os.path.join(load_dir, f"{model_type}_boroondara.h5")

            # If not found, try alternative naming patterns
            if not os.path.exists(model_path):
                alt_path = os.path.join(load_dir, f"{model_type}_all_sites.h5")
                if os.path.exists(alt_path):
                    print(f"Found alternative model file {alt_path}")
                    model_path = alt_path
                else:
                    print(f"Warning: Model file {model_path} does not exist")
                    continue

        try:
            # Define custom objects for compatibility with different Keras versions
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mean_squared_error': tf.keras.losses.MeanSquaredError(),
                'mape': tf.keras.metrics.MeanAbsolutePercentageError(),
                'PositionalEncoding': PositionalEncoding
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


def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    args: Any,
    site_id: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Train a model.

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
        Tuple of (trained model, training results)
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

    # Add training parameters to config
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs

    # Create model
    model = create_model(model_type, config)

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
    callbacks = create_callbacks(
        model_type,
        args.patience,
        args.log_dir,
        site_id
    )

    # Reshape input for RNN models if needed
    if model_type in ['lstm', 'gru', 'transformer']:
        X_train_model = reshape_for_rnn(X_train)
        X_test_model = reshape_for_rnn(X_test)
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

    # Save training history
    if site_id:
        history_dir = os.path.join(args.results_dir, "training_history", "per_site", site_id)
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, f"{model_type}_history.csv")
    else:
        history_dir = os.path.join(args.results_dir, "training_history")
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, f"{model_type}_history.csv")

    import pandas as pd
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")

    return model, {
        'model_type': model_type,
        'loss': loss,
        'mape': mape,
        'model_path': model_path,
        'history': history.history
    }


def predict_with_models(
    models: Dict[str, tf.keras.Model],
    X_test: np.ndarray
) -> Dict[str, np.ndarray]:
    """Generate predictions using multiple models.

    Args:
        models: Dictionary of loaded models
        X_test: Test features

    Returns:
        Dictionary mapping model types to predictions
    """
    predictions = {}

    for model_type, model in models.items():
        # Reshape input for RNN models if needed
        if model_type in ['lstm', 'gru', 'transformer']:
            X_test_reshaped = reshape_for_rnn(X_test)
        else:
            X_test_reshaped = X_test

        # Generate predictions
        predictions[model_type] = model.predict(X_test_reshaped)

    return predictions

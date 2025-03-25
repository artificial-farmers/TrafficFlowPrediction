"""
Training script for Boroondara traffic flow prediction models.
"""
from data.boroondara_preprocessing import prepare_boroondara_dataset
from src.models.saes import SAEModel
from src.models.gru import GRUModel
from src.models.lstm import LSTMModel
from config.boroondara_config import get_all_configs
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_tensorflow() -> None:
    """
    Configure TensorFlow settings.
    """
    # Set TensorFlow log level
    tf.get_logger().setLevel('ERROR')

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)


def load_boroondara_data(data_dir: str) -> tuple:
    """
    Load preprocessed Boroondara dataset.

    Args:
        data_dir: Directory containing preprocessed data files

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    return X_train, y_train, X_test, y_test


def prepare_data_if_needed(
    scats_data_path: str,
    metadata_path: str,
    processed_dir: str,
    force_preprocess: bool = False
) -> tuple:
    """
    Prepare or load the Boroondara dataset.

    Args:
        scats_data_path: Path to SCATS data CSV file
        metadata_path: Path to SCATS metadata CSV file
        processed_dir: Directory to save/load processed data
        force_preprocess: Whether to force preprocessing even if data exists

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir, exist_ok=True)

    X_train_path = os.path.join(processed_dir, 'X_train.npy')

    # Check if processed data already exists
    if not force_preprocess and os.path.exists(X_train_path):
        print("Loading preprocessed data...")
        return load_boroondara_data(processed_dir)

    # Process data
    print("Processing raw data...")
    prepare_boroondara_dataset(scats_data_path, metadata_path, processed_dir)

    return load_boroondara_data(processed_dir)


def create_model(model_type: str, config: Dict[str, Any]) -> Union[LSTMModel, GRUModel, SAEModel]:
    """
    Create a model instance based on the specified type.

    Args:
        model_type: Type of model to create ('lstm', 'gru', or 'saes')
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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_callbacks(
    config: Dict[str, Any],
    model_type: str,
    log_dir: str
) -> list:
    """
    Create training callbacks.

    Args:
        config: Model configuration dictionary
        model_type: Type of model being trained
        log_dir: Directory for TensorBoard logs

    Returns:
        List of callbacks
    """
    callbacks = []

    # Early stopping
    patience = config.get('early_stopping_patience', 10)
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    )

    # TensorBoard logging
    model_log_dir = os.path.join(
        log_dir, model_type, datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=model_log_dir,
            histogram_freq=1,
            write_graph=True
        )
    )

    # Model checkpoint
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_log_dir, 'checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    )

    return callbacks


def reshape_input_for_rnn(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Reshape input data for recurrent neural networks (LSTM, GRU).

    Args:
        X_train: Training features
        X_test: Testing features

    Returns:
        Tuple of reshaped (X_train, X_test)
    """
    # Reshape to (samples, time steps, features)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train_reshaped, X_test_reshaped


def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    output_dir: str,
    log_dir: str
) -> Dict[str, Any]:
    """
    Train a model with the given data.

    Args:
        model_type: Type of model to train ('lstm', 'gru', or 'saes')
        X_train: Training features
        y_train: Training targets
        X_test: Testing features
        y_test: Testing targets
        config: Model configuration
        output_dir: Directory to save trained models
        log_dir: Directory for TensorBoard logs

    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*50}")

    # Create model
    model = create_model(model_type, config)

    # Build model architecture
    model.build()

    # For SAE, perform pretraining if needed
    if model_type == 'saes':
        print("\nPre-training SAE layers...")
        model.pretrain(
            X_train,
            y_train,
            validation_split=config.get('validation_split', 0.1),
            epochs=config.get('pretraining_epochs', 50)
        )

    # Compile model
    model.compile()

    # Create callbacks
    callbacks = create_callbacks(config, model_type, log_dir)

    # Reshape input for RNN models
    if model_type in ['lstm', 'gru']:
        X_train, X_test = reshape_input_for_rnn(X_train, X_test)

    # Train model
    print("\nTraining model...")
    history = model.train(
        X_train,
        y_train,
        validation_split=config.get('validation_split', 0.1),
        callbacks=callbacks
    )

    # Save model
    model_path = os.path.join(output_dir, f"{model_type}_boroondara.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(
        output_dir, f"{model_type}_boroondara_history.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")

    # Evaluate model
    print("\nEvaluating model...")
    evaluation = model.model.evaluate(X_test, y_test, verbose=1)

    # Return results
    results = {
        'model_type': model_type,
        'loss': evaluation[0],
        'mape': evaluation[1],
        'model_path': model_path,
        'history_path': history_csv_path
    }

    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Test MAPE: {results['mape']:.4f}")

    return results


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train traffic flow prediction models on Boroondara data")

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to train: 'lstm', 'gru', 'saes', or 'all'"
    )

    parser.add_argument(
        "--scats_data",
        type=str,
        default="data/raw/Scats Data October 2006.csv",
        help="Path to SCATS data CSV file"
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="data/raw/SCATSSiteListingSpreadsheet_VicRoads.csv",
        help="Path to SCATS metadata CSV file"
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed/boroondara",
        help="Directory for processed data"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models/boroondara",
        help="Directory to save trained models"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/boroondara",
        help="Directory for TensorBoard logs"
    )

    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Force preprocessing of data even if already exists"
    )

    return parser.parse_args()


def main():
    """
    Main function to train Boroondara traffic flow prediction models.
    """
    # Parse arguments
    args = parse_args()

    # Setup TensorFlow
    setup_tensorflow()

    # Create output directories
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Get model configurations
    all_configs = get_all_configs()

    # Determine models to train
    if args.models == 'all':
        models_to_train = ['lstm', 'gru', 'saes']
    else:
        models_to_train = [model.strip() for model in args.models.split(',')]

    # Prepare or load data
    X_train, y_train, X_test, y_test = prepare_data_if_needed(
        args.scats_data,
        args.metadata,
        args.processed_dir,
        args.force_preprocess
    )

    print("\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    # Train models
    results = []

    for model_type in models_to_train:
        if model_type not in all_configs:
            print(f"Warning: Unknown model type '{model_type}'. Skipping.")
            continue

        config = all_configs[model_type]

        result = train_model(
            model_type,
            X_train,
            y_train,
            X_test,
            y_test,
            config,
            args.output_dir,
            args.log_dir
        )

        results.append(result)

    # Print summary
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)

    for result in results:
        print(f"\n{result['model_type'].upper()}:")
        print(f"  Test Loss: {result['loss']:.4f}")
        print(f"  Test MAPE: {result['mape']:.4f}")
        print(f"  Model saved: {result['model_path']}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

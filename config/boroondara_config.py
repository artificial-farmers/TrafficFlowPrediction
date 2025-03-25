"""
Configuration settings for Boroondara traffic flow prediction models.
"""
from typing import Dict, Any


def get_lstm_config() -> Dict[str, Any]:
    """
    Get configuration for LSTM model.

    Returns:
        Dictionary with model configuration
    """
    return {
        "input_dim": 12,  # 12 time steps (3 hours of 15-minute intervals)
        "lstm_units": [64, 64],
        "dropout_rate": 0.2,
        "batch_size": 256,
        "epochs": 100,
        "validation_split": 0.1,
        "learning_rate": 0.001,
        "early_stopping_patience": 10
    }


def get_gru_config() -> Dict[str, Any]:
    """
    Get configuration for GRU model.

    Returns:
        Dictionary with model configuration
    """
    return {
        "input_dim": 12,  # 12 time steps (3 hours of 15-minute intervals)
        "gru_units": [64, 64],
        "dropout_rate": 0.2,
        "batch_size": 256,
        "epochs": 100,
        "validation_split": 0.1,
        "learning_rate": 0.001,
        "early_stopping_patience": 10
    }


def get_sae_config() -> Dict[str, Any]:
    """
    Get configuration for SAE model.

    Returns:
        Dictionary with model configuration
    """
    return {
        "input_dim": 12,  # 12 time steps (3 hours of 15-minute intervals)
        "hidden_dims": [400, 400, 400],
        "dropout_rate": 0.2,
        "batch_size": 256,
        "epochs": 100,
        "pretraining_epochs": 50,
        "validation_split": 0.1,
        "learning_rate": 0.001,
        "early_stopping_patience": 10
    }


def get_all_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for all model types.

    Returns:
        Dictionary mapping model names to their configurations
    """
    return {
        "lstm": get_lstm_config(),
        "gru": get_gru_config(),
        "saes": get_sae_config()
    }

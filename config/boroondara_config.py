"""
Configuration settings for Boroondara traffic flow prediction models.
"""
from typing import Dict, Any

def get_lstm_config() -> Dict[str, Any]:
    """Get LSTM model configuration.

    Returns:
        Dictionary with LSTM model configuration
    """
    return {
        "input_dim": 12,  # Sequence length (number of time steps)
        "lstm_units": [64, 64],  # Units in LSTM layers
        "dropout_rate": 0.2,  # Dropout rate
        "early_stopping_patience": 10,  # Early stopping patience
    }

def get_gru_config() -> Dict[str, Any]:
    """Get GRU model configuration.

    Returns:
        Dictionary with GRU model configuration
    """
    return {
        "input_dim": 12,  # Sequence length (number of time steps)
        "gru_units": [64, 64],  # Units in GRU layers
        "dropout_rate": 0.2,  # Dropout rate
        "early_stopping_patience": 10,  # Early stopping patience
    }

def get_saes_config() -> Dict[str, Any]:
    """Get SAE model configuration.

    Returns:
        Dictionary with SAE model configuration
    """
    return {
        "input_dim": 12,  # Sequence length (number of time steps)
        "hidden_dims": [400, 400, 400],  # Dimensions of hidden layers
        "dropout_rate": 0.2,  # Dropout rate
        "pretraining_epochs": 50,  # Epochs for pretraining
        "early_stopping_patience": 10,  # Early stopping patience
    }

def get_transformer_config() -> Dict[str, Any]:
    """Get Transformer model configuration.

    Returns:
        Dictionary with Transformer model configuration
    """
    return {
        "input_dim": 12,  # Sequence length (number of time steps)
        "d_model": 64,  # Model dimension
        "num_heads": 4,  # Number of attention heads
        "dff": 128,  # Feed-forward network dimension
        "num_layers": 2,  # Number of transformer layers
        "dropout_rate": 0.1,  # Dropout rate
        "early_stopping_patience": 10,  # Early stopping patience
    }

def get_all_configs() -> Dict[str, Dict[str, Any]]:
    """Get all model configurations.

    Returns:
        Dictionary with configurations for all models
    """
    return {
        "lstm": get_lstm_config(),
        "gru": get_gru_config(),
        "saes": get_saes_config(),
        "transformer": get_transformer_config(),
    }

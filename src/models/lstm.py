"""
LSTM model implementation for traffic flow prediction.
"""
from typing import Dict, List, Tuple, Union

import tensorflow as tf

from src.models.base import BaseModel


class LSTMModel(BaseModel):
    """LSTM model for traffic flow prediction."""

    def __init__(self, config: Dict):
        """Initialize the LSTM model.

        Args:
            config: Dictionary containing model configuration.
                - input_dim: Input dimension
                - lstm_units: List of units for LSTM layers
                - dropout_rate: Dropout rate (default: 0.2)
        """
        super().__init__(config)
        self.input_dim = config.get("input_dim", 12)
        self.lstm_units = config.get("lstm_units", [64, 64])
        self.dropout_rate = config.get("dropout_rate", 0.2)

    def build(self) -> None:
        """Build the LSTM model architecture."""
        self.model = tf.keras.Sequential()

        # First LSTM layer with return sequences
        self.model.add(
            tf.keras.layers.LSTM(
                self.lstm_units[0],
                input_shape=(self.input_dim, 1),
                return_sequences=True,
            )
        )

        # Second LSTM layer
        self.model.add(tf.keras.layers.LSTM(self.lstm_units[1]))

        # Dropout for regularization
        self.model.add(tf.keras.layers.Dropout(self.dropout_rate))

        # Output layer
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

"""
GRU model implementation for traffic flow prediction.
"""
from typing import Dict, List, Tuple, Union

import tensorflow as tf

from src.models.base import BaseModel


class GRUModel(BaseModel):
    """GRU model for traffic flow prediction."""

    def __init__(self, config: Dict):
        """Initialize the GRU model.

        Args:
            config: Dictionary containing model configuration.
                - input_dim: Input dimension
                - gru_units: List of units for GRU layers
                - dropout_rate: Dropout rate (default: 0.2)
        """
        super().__init__(config)
        self.input_dim = config.get("input_dim", 12)
        self.gru_units = config.get("gru_units", [64, 64])
        self.dropout_rate = config.get("dropout_rate", 0.2)

    def build(self) -> None:
        """Build the GRU model architecture."""
        self.model = tf.keras.Sequential()

        # First GRU layer with return sequences
        self.model.add(
            tf.keras.layers.GRU(
                self.gru_units[0],
                input_shape=(self.input_dim, 1),
                return_sequences=True,
            )
        )

        # Second GRU layer
        self.model.add(tf.keras.layers.GRU(self.gru_units[1]))

        # Dropout for regularization
        self.model.add(tf.keras.layers.Dropout(self.dropout_rate))

        # Output layer
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

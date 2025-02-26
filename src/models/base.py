"""
Base model interface for traffic flow prediction models.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import tensorflow as tf


class BaseModel(ABC):
    """Base class for all traffic flow prediction models."""

    def __init__(self, config: Dict):
        """Initialize the model with configuration parameters.

        Args:
            config: Dictionary containing model configuration.
        """
        self.config = config
        self.model = None

    @abstractmethod
    def build(self) -> None:
        """Build the model architecture."""
        pass

    def compile(self) -> None:
        """Compile the model with optimizer and loss function."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")

        self.model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            metrics=["mape"],
        )

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.05,
        callbacks: Optional[List] = None,
    ) -> tf.keras.callbacks.History:
        """Train the model.

        Args:
            x_train: Training input data.
            y_train: Training target data.
            validation_split: Fraction of data to use for validation.
            callbacks: List of Keras callbacks.

        Returns:
            Training history.
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")

        return self.model.fit(
            x_train,
            y_train,
            batch_size=self.config.get("batch_size", 256),
            epochs=self.config.get("epochs", 100),
            validation_split=validation_split,
            callbacks=callbacks or [],
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions with the model.

        Args:
            x: Input data.

        Returns:
            Model predictions.
        """
        if self.model is None:
            raise ValueError("Model must be built before prediction")

        return self.model.predict(x)

    def save(self, filepath: str) -> None:
        """Save the model to disk.

        Args:
            filepath: Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")

        self.model.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> "BaseModel":
        """Load a model from disk.

        Args:
            filepath: Path to the saved model.

        Returns:
            Loaded model.
        """
        instance = cls({})  # Create a base instance
        instance.model = tf.keras.models.load_model(filepath)
        return instance

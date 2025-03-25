"""
Stacked Autoencoder (SAE) model implementation for traffic flow prediction.
"""
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import tensorflow as tf

from src.models.base import BaseModel


class SAEModel(BaseModel):
    """Stacked Autoencoder model for traffic flow prediction."""

    def __init__(self, config: Dict):
        """Initialize the SAE model.

        Args:
            config: Dictionary containing model configuration.
                - input_dim: Input dimension
                - hidden_dims: List of hidden layer dimensions
                - dropout_rate: Dropout rate (default: 0.2)
        """
        super().__init__(config)
        self.input_dim = config.get("input_dim", 12)
        self.hidden_dims = config.get("hidden_dims", [400, 400, 400])
        self.dropout_rate = config.get("dropout_rate", 0.2)
        self.encoders = []
        self.sae_models = []

    def build(self) -> None:
        """Build the SAE model architecture."""
        # Create the stacked model (final model)
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Dense(
                self.hidden_dims[0], input_dim=self.input_dim, name="hidden1"
            )
        )
        self.model.add(tf.keras.layers.Activation("sigmoid"))
        self.model.add(tf.keras.layers.Dense(
            self.hidden_dims[1], name="hidden2"))
        self.model.add(tf.keras.layers.Activation("sigmoid"))
        self.model.add(tf.keras.layers.Dense(
            self.hidden_dims[2], name="hidden3"))
        self.model.add(tf.keras.layers.Activation("sigmoid"))
        self.model.add(tf.keras.layers.Dropout(self.dropout_rate))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    def pretrain(
        self, x_train: np.ndarray, y_train: np.ndarray, validation_split: float = 0.05, epochs: Optional[int] = None
    ) -> None:
        """Pretrain the SAE model layer by layer.

        Args:
            x_train: Training input data
            y_train: Training target data
            validation_split: Fraction of data to use for validation
            epochs: Number of epochs for pretraining (overrides config)
        """
        # Use provided epochs or get from config
        pretraining_epochs = epochs if epochs is not None else self.config.get(
            "pretraining_epochs", 50)
        batch_size = self.config.get("batch_size", 256)

        # Create three separate autoencoders with proper dimensions
        # Need to create these here rather than in build() to ensure proper dimensions

        # First autoencoder: 12 -> 400 -> 1
        print("Pre-training autoencoder 1/3")
        ae1 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hidden_dims[0], input_dim=self.input_dim, name="hidden"),
            tf.keras.layers.Activation("sigmoid"),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        ae1.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            metrics=["mape"]
        )

        ae1.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=pretraining_epochs,
            validation_split=validation_split,
            verbose=1
        )

        # Create encoder from first autoencoder to transform data
        encoder1 = tf.keras.Sequential()
        encoder1.add(tf.keras.layers.Dense(
            self.hidden_dims[0],
            input_dim=self.input_dim,
            activation='sigmoid'
        ))
        # Copy weights from trained layer to encoder
        encoder1.layers[0].set_weights(ae1.layers[0].get_weights())

        # Transform data through first encoder
        encoded_input = encoder1.predict(x_train)

        # Second autoencoder: 400 -> 400 -> 1
        print("Pre-training autoencoder 2/3")
        ae2 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hidden_dims[1], input_dim=self.hidden_dims[0], name="hidden"),
            tf.keras.layers.Activation("sigmoid"),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        ae2.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            metrics=["mape"]
        )

        ae2.fit(
            encoded_input,
            y_train,
            batch_size=batch_size,
            epochs=pretraining_epochs,
            validation_split=validation_split,
            verbose=1
        )

        # Create encoder from second autoencoder
        encoder2 = tf.keras.Sequential()
        encoder2.add(tf.keras.layers.Dense(
            self.hidden_dims[1],
            input_dim=self.hidden_dims[0],
            activation='sigmoid'
        ))
        # Copy weights from trained layer to encoder
        encoder2.layers[0].set_weights(ae2.layers[0].get_weights())

        # Transform data through second encoder
        encoded_input2 = encoder2.predict(encoded_input)

        # Third autoencoder: 400 -> 400 -> 1
        print("Pre-training autoencoder 3/3")
        ae3 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hidden_dims[2], input_dim=self.hidden_dims[1], name="hidden"),
            tf.keras.layers.Activation("sigmoid"),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        ae3.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            metrics=["mape"]
        )

        ae3.fit(
            encoded_input2,
            y_train,
            batch_size=batch_size,
            epochs=pretraining_epochs,
            validation_split=validation_split,
            verbose=1
        )

        # Transfer weights to the stacked model
        self.model.get_layer('hidden1').set_weights(
            ae1.layers[0].get_weights())
        self.model.get_layer('hidden2').set_weights(
            ae2.layers[0].get_weights())
        self.model.get_layer('hidden3').set_weights(
            ae3.layers[0].get_weights())

        print("Pre-training complete. Weights transferred to stacked model.")

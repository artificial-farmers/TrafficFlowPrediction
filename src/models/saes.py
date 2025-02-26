"""
Stacked Autoencoder (SAE) model implementation for traffic flow prediction.
"""
from typing import Dict, List, Tuple, Union

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

        # Individual SAEs for pre-training
        self.sae_models = []

    def _build_single_autoencoder(
        self, input_dim: int, hidden_dim: int, output_dim: int
    ) -> tf.keras.Model:
        """Build a single autoencoder.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension

        Returns:
            A single autoencoder model
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(hidden_dim, input_dim=input_dim, name="hidden"))
        model.add(tf.keras.layers.Activation("sigmoid"))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(output_dim, activation="sigmoid"))

        return model

    def build(self) -> None:
        """Build the SAE model architecture."""
        # Create individual autoencoders for pretraining
        self.sae_models = [
            self._build_single_autoencoder(self.input_dim, self.hidden_dims[0], 1),
            self._build_single_autoencoder(self.hidden_dims[0], self.hidden_dims[1], 1),
            self._build_single_autoencoder(self.hidden_dims[1], self.hidden_dims[2], 1),
        ]

        # Create the stacked model
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Dense(
                self.hidden_dims[0], input_dim=self.input_dim, name="hidden1"
            )
        )
        self.model.add(tf.keras.layers.Activation("sigmoid"))
        self.model.add(tf.keras.layers.Dense(self.hidden_dims[1], name="hidden2"))
        self.model.add(tf.keras.layers.Activation("sigmoid"))
        self.model.add(tf.keras.layers.Dense(self.hidden_dims[2], name="hidden3"))
        self.model.add(tf.keras.layers.Activation("sigmoid"))
        self.model.add(tf.keras.layers.Dropout(self.dropout_rate))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    def pretrain(
        self, x_train: np.ndarray, y_train: np.ndarray, validation_split: float = 0.05
    ) -> None:
        """Pretrain the SAE model layer by layer.

        Args:
            x_train: Training input data
            y_train: Training target data
            validation_split: Fraction of data to use for validation
        """
        temp_input = x_train

        # Train each autoencoder
        for i, sae_model in enumerate(self.sae_models):
            print(f"Pre-training autoencoder {i+1}/{len(self.sae_models)}")

            if i > 0:
                # Get the output of the previous hidden layer
                prev_model = self.sae_models[i-1]
                hidden_layer_model = tf.keras.Model(
                    inputs=prev_model.input,
                    outputs=prev_model.get_layer("hidden").output
                )
                temp_input = hidden_layer_model.predict(temp_input)

            # Compile and train the current autoencoder
            sae_model.compile(
                loss="mse",
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                metrics=["mape"]
            )

            sae_model.fit(
                temp_input,
                y_train,
                batch_size=self.config.get("batch_size", 256),
                epochs=self.config.get("epochs", 100),
                validation_split=validation_split,
                verbose=1,
            )

        # Transfer weights to the stacked model
        for i, sae_model in enumerate(self.sae_models):
            weights = sae_model.get_layer("hidden").get_weights()
            self.model.get_layer(f"hidden{i+1}").set_weights(weights)

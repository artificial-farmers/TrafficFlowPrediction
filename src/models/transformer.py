"""
Transformer model implementation for traffic flow prediction.
"""
import tensorflow as tf
from typing import Dict, List, Tuple, Union

from src.models.base import BaseModel


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for transformer model."""

    def __init__(self, position: int, d_model: int, **kwargs):
        """Initialize positional encoding.

        Args:
            position: Maximum sequence length
            d_model: Dimensionality of the model
            **kwargs: Additional layer parameters
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position: tf.Tensor, i: tf.Tensor, d_model: int) -> tf.Tensor:
        """Calculate angles for positional encoding.

        Args:
            position: Position tensor
            i: Dimension tensor
            d_model: Dimensionality of the model

        Returns:
            Angle tensor
        """
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        """Create positional encoding.

        Args:
            position: Maximum sequence length
            d_model: Dimensionality of the model

        Returns:
            Positional encoding tensor
        """
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model
        )

        # Apply sine to even indices in the array
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Apply cosine to odd indices in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Alternate sine and cosine values
        pos_encoding = tf.concat([sines, cosines], axis=-1)

        # Add a batch dimension
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply positional encoding to inputs.

        Args:
            inputs: Input tensor

        Returns:
            Encoded tensor
        """
        # inputs shape: (batch_size, seq_len, d_model)
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Dictionary with layer configuration
        """
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model
        })
        return config


class TransformerModel(BaseModel):
    """Transformer model for traffic flow prediction."""

    def __init__(self, config: Dict):
        """Initialize the Transformer model.

        Args:
            config: Dictionary containing model configuration.
                - input_dim: Input dimension (sequence length)
                - d_model: Dimensionality of the model
                - num_heads: Number of attention heads
                - dff: Dimensionality of feed forward network
                - num_layers: Number of transformer layers
                - dropout_rate: Dropout rate (default: 0.1)
        """
        super().__init__(config)
        self.input_dim = config.get("input_dim", 12)
        self.d_model = config.get("d_model", 64)
        self.num_heads = config.get("num_heads", 4)
        self.dff = config.get("dff", 128)
        self.num_layers = config.get("num_layers", 2)
        self.dropout_rate = config.get("dropout_rate", 0.1)

    def build(self) -> None:
        """Build the Transformer model architecture."""
        # Input layer
        inputs = tf.keras.layers.Input(shape=(self.input_dim, 1))

        # Reshape and project inputs to d_model dimension
        x = tf.keras.layers.Reshape((self.input_dim, 1))(inputs)
        x = tf.keras.layers.Dense(self.d_model, activation='relu')(x)

        # Add positional encoding
        x = PositionalEncoding(self.input_dim, self.d_model)(x)

        # Transformer encoder layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads
            )(x, x, x)

            # Add & Norm
            attn_output = tf.keras.layers.Dropout(self.dropout_rate)(attn_output)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

            # Feed Forward Network
            ffn_output = tf.keras.Sequential([
                tf.keras.layers.Dense(self.dff, activation='relu'),
                tf.keras.layers.Dense(self.d_model)
            ])(x)

            # Add & Norm
            ffn_output = tf.keras.layers.Dropout(self.dropout_rate)(ffn_output)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Global average pooling to reduce sequence dimension
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Output layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        # Create the model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

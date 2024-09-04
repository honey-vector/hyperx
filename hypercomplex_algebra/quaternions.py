import tensorflow as tf
from keras.layers import Layer
from typing import Dict, Any

class QuaternionDense(Layer):
    def __init__(self, units: int, use_bias: bool = True, **kwargs):
        """
        Initialize the QuaternionDense layer.

        Args:
            units (int): The number of output units.
            use_bias (bool): Whether to use a bias term. Default is True.
            **kwargs: Additional keyword arguments for the base Layer class.
        """
        super(QuaternionDense, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape: tf.TensorShape):
        """
        Build the layer weights.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        assert input_shape[-1] % 4 == 0, "Input last dimension must be divisible by 4"
        input_dim = input_shape[-1] // 4
        kernel_shape = (input_dim, self.units)
        
        self.kernel_r = self.add_weight(shape=kernel_shape,
                                        initializer='glorot_uniform',
                                        name='kernel_r')
        self.kernel_i = self.add_weight(shape=kernel_shape,
                                        initializer='glorot_uniform',
                                        name='kernel_i')
        self.kernel_j = self.add_weight(shape=kernel_shape,
                                        initializer='glorot_uniform',
                                        name='kernel_j')
        self.kernel_k = self.add_weight(shape=kernel_shape,
                                        initializer='glorot_uniform',
                                        name='kernel_k')
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(4 * self.units,),
                                        initializer='zeros',
                                        name='bias')
        
        super(QuaternionDense, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform the quaternion multiplication.

        Args:
            inputs (tf.Tensor): Input tensor of shape (..., 4*input_dim).

        Returns:
            tf.Tensor: Output tensor of shape (..., 4*units).
        """
        if inputs.shape[-1] % 4 != 0:
            raise ValueError("Input last dimension must be divisible by 4")

        r, i, j, k = tf.split(inputs, num_or_size_splits=4, axis=-1)

        # Quaternion multiplication
        or_ = r @ self.kernel_r - i @ self.kernel_i - j @ self.kernel_j - k @ self.kernel_k
        oi = r @ self.kernel_i + i @ self.kernel_r + j @ self.kernel_k - k @ self.kernel_j
        oj = r @ self.kernel_j - i @ self.kernel_k + j @ self.kernel_r + k @ self.kernel_i
        ok = r @ self.kernel_k + i @ self.kernel_j - j @ self.kernel_i + k @ self.kernel_r

        output = tf.concat([or_, oi, oj, ok], axis=-1)
        
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        return output

    def get_config(self) -> Dict[str, Any]:
        """
        Get the config dictionary for the layer.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        config = super(QuaternionDense, self).get_config()
        config.update({'units': self.units, 'use_bias': self.use_bias})
        return config

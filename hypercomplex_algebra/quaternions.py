import tensorflow as tf
from tensorflow.keras.layers import Layer

class QuaternionDense(Layer):
    def __init__(self, units, **kwargs):
        super(QuaternionDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # input_shape[-1] should be divisible by 4 for quaternions
        assert input_shape[-1] % 4 == 0, "Input size must be divisible by 4"
        self.w = self.add_weight(shape=(input_shape[-1] // 4, self.units * 4),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        # Split input into quaternion components
        w, x, y, z = tf.split(inputs, num_or_size_splits=4, axis=-1)

        # Reshape weights for proper quaternion multiplication
        W_w, W_x, W_y, W_z = tf.split(self.w, num_or_size_splits=4, axis=-1)

        # Perform full quaternion multiplication
        new_w = tf.matmul(w, W_w) - tf.matmul(x, W_x) - tf.matmul(y, W_y) - tf.matmul(z, W_z)
        new_x = tf.matmul(w, W_x) + tf.matmul(x, W_w) + tf.matmul(y, W_z) - tf.matmul(z, W_y)
        new_y = tf.matmul(w, W_y) - tf.matmul(x, W_z) + tf.matmul(y, W_w) + tf.matmul(z, W_x)
        new_z = tf.matmul(w, W_z) + tf.matmul(x, W_y) - tf.matmul(y, W_x) + tf.matmul(z, W_w)

        # Concatenate the results into the final output
        return tf.concat([new_w, new_x, new_y, new_z], axis=-1)

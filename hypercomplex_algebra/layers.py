import tensorflow as tf
from tensorflow.keras.layers import Layer

class QuaternionDense(Layer):
    def __init__(self, units, **kwargs):
        super(QuaternionDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1]//4, self.units * 4),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        # Split input into quaternions
        w, x, y, z = tf.split(inputs, num_or_size_splits=4, axis=-1)
        # Perform quaternion multiplication
        # This is a simplified operation, you'll need to implement full quaternion multiplication
        output_w = tf.matmul(w, self.w)
        output_x = tf.matmul(x, self.w)
        output_y = tf.matmul(y, self.w)
        output_z = tf.matmul(z, self.w)
        return tf.concat([output_w, output_x, output_y, output_z], axis=-1)

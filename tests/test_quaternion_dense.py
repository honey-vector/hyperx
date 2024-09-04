import pytest
import tensorflow as tf
import numpy as np
from hypercomplex_algebra.quaternions import QuaternionDense

# Add a small epsilon for floating-point comparisons
EPSILON = 1e-6

# Test: Correct output shape
def test_quaternion_dense_output_shape():
    layer = QuaternionDense(units=8)
    input_data = np.random.rand(10, 16).astype(np.float32)
    output_data = layer(input_data)

    assert output_data.shape == (10, 32), "Output shape mismatch"

# Test: Zero input
def test_quaternion_dense_zero_input():
    layer = QuaternionDense(units=4)
    input_data = tf.zeros((5, 16))
    output_data = layer(input_data)

    assert not tf.reduce_all(tf.abs(output_data) < 1e-6), "Output should not be all zeros for zero input"

# Test: Quaternion multiplication properties
def test_quaternion_dense_multiplication():
    layer = QuaternionDense(units=1, use_bias=False)
    layer.build((None, 4))
    
    # Set weights to identity quaternion
    identity_quaternion = np.eye(4).astype(np.float32)
    layer.set_weights([
        identity_quaternion[0:1, :],  # kernel_r
        identity_quaternion[1:2, :],  # kernel_i
        identity_quaternion[2:3, :],  # kernel_j
        identity_quaternion[3:4, :]   # kernel_k
    ])
    
    # Test with unit quaternions
    input_data = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32)
    output = layer(input_data)
    
    expected_output = input_data
    assert np.allclose(output.numpy(), expected_output, atol=1e-6), "Quaternion multiplication failed for unit quaternions"

# Test: Non-divisible input dimensions should raise an error
def test_quaternion_dense_invalid_input_shape():
    layer = QuaternionDense(units=4)
    invalid_input_data = tf.random.normal((5, 18))
    with pytest.raises(ValueError, match="Input last dimension must be divisible by 4"):
        layer(invalid_input_data)

# Test: Serialization and deserialization
def test_quaternion_dense_serialization():
    layer = QuaternionDense(units=8)
    config = layer.get_config()
    new_layer = QuaternionDense.from_config(config)
    assert layer.units == new_layer.units, "Serialization/deserialization failed"

# Test: Training with gradient tape
def test_quaternion_dense_training():
    layer = QuaternionDense(units=4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = layer(inputs)
            loss = tf.reduce_mean(tf.square(predictions - targets))
        gradients = tape.gradient(loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, layer.trainable_variables))
        return loss
    
    inputs = tf.random.normal((32, 16))
    targets = tf.random.normal((32, 16))
    
    initial_loss = train_step(inputs, targets)
    for _ in range(10):
        loss = train_step(inputs, targets)
    
    assert loss < initial_loss, "Training did not reduce the loss"

if __name__ == "__main__":
    pytest.main()

# Hyperx Algebra

A library for general hypercomplex algebras with deep learning integration.


## Overview

The **Hyperx** is an open-source Python library designed to support neural network layers that operate over general hypercomplex algebras, including quaternions, octonions (WIP), and beyond (WIP). This library provides integration with Keras, allowing the creation of dense and convolutional layers that leverage the mathematical properties of hypercomplex numbers.


## Features

- **General Hypercomplex Algebra Support:** Base classes for hypercomplex numbers like quaternions, octonions, etc.
- **Keras Integration:** Custom Keras layers (`QuaternionDense`, etc.) that support hypercomplex operations.
- **Modular and Extensible:** Easily extendable to include other hypercomplex numbers or algebras.


## Installation

You can install the library via `pip`:

```
pip install hyperx
```

Alternatively, you can clone the repository and install it manually:

```
git clone https://github.com/honey-vector/hyperx.git
cd hyperx
pip install .
```


## Usage
Here is a basic example of using a quaternion dense layer in a Keras model:

```
import tensorflow as tf
from hypercomplex_algebra.layers import QuaternionDense

model = tf.keras.Sequential([
    QuaternionDense(10, input_shape=(16,)),
    tf.keras.layers.Activation('relu'),
    QuaternionDense(10),
])

model.summary()
```


## License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue)


## Contact
For any questions or inquiries, please contact Renato Boemer.

**# deeplearning-in-medicine
A python project on medical science**

Let's understand the basics first.

**1. What is keras?**

Keras is an open-source neural network library written in Python. It is designed to be user-friendly, modular, and extensible. Keras allows for easy and fast prototyping of deep learning models by providing a high-level, intuitive API that abstracts away many of the complexities of building and training neural networks.

**Explaining "from keras.layers import Input, Lambda, Dense, Flatten"
**
**Input:** This class is used to instantiate a Keras tensor. It represents the input to the neural network model. Typically, you use it to define the shape and type of the input data.

**Lambda:** This class allows you to wrap arbitrary expressions as a Layer object. It's useful for performing custom transformations or computations on the input or output of a neural network layer.

**Dense:** This class represents a fully connected (or densely connected) layer in a neural network. It connects every neuron in the previous layer to every neuron in the current layer.

**Flatten:** This class is used to flatten the input, which means it reshapes the input tensor into a one-dimensional array. It's commonly used to transition from convolutional layers (which produce 3D outputs) to fully connected layers (which require 1D inputs).

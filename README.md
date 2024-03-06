**# deeplearning-in-medicine
A python project on medical science**

Let's understand the basics first.

**1. What is keras? Add more details to the sections of the code used for the following project.**

Keras is an open-source neural network library written in Python. It is designed to be user-friendly, modular, and extensible. Keras allows for easy and fast prototyping of deep learning models by providing a high-level, intuitive API that abstracts away many of the complexities of building and training neural networks.

a) from keras.layers import Input, Lambda, Dense, Flatten: This line imports specific classes and functions from the keras.layers module. It imports Input, Lambda, Dense, and Flatten which are commonly used building blocks for defining neural network architectures in Keras.

b) from keras.models import Model: This line imports the Model class from the keras.models module. The Model class is used to instantiate a Keras model, which represents the entire neural network architecture, including its input and output layers.

c) from keras.applications.vgg16 import VGG16: This line imports the VGG16 pre-trained deep learning model from the keras.applications.vgg16 module. VGG16 is a popular convolutional neural network architecture that has been pre-trained on the ImageNet dataset.

d) from keras.applications.vgg16 import preprocess_input: This line imports the preprocess_input function from the keras.applications.vgg16 module. This function preprocesses input images according to the requirements of the VGG16 model.

e) from keras.preprocessing import image: This line imports the image module from the keras.preprocessing package. This module provides utilities for loading and preprocessing image data.

f) from keras.preprocessing.image import ImageDataGenerator: This line imports the ImageDataGenerator class from the keras.preprocessing.image module. The ImageDataGenerator class is used for real-time data augmentation and preprocessing of image data during training.

g) from keras.models import Sequential: This line imports the Sequential class from the keras.models module. The Sequential class is used to create a linear stack of layers for building neural network models.

h) import numpy as np: This line imports the numpy library and aliases it as np. NumPy is a popular library for numerical computing in Python, and it is commonly used for handling arrays and matrices.

i) from glob import glob: This line imports the glob function from the glob module. The glob function is used to find all the pathnames matching a specified pattern according to the rules used by the Unix shell.

j) import matplotlib.pyplot as plt: This line imports the pyplot module from the matplotlib library and aliases it as plt. Matplotlib is a plotting library for Python, and pyplot is a collection of functions that make matplotlib work like MATLAB. It is commonly used for data visualization.


**2. Let's leard more about VGG16 Model**

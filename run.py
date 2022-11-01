#!/bin/python3

# =====================
# IMPORTS
# =====================
# numpy for all of the maths
import numpy as np
# random and default_rng for shuffling the data
from numpy import random

# pandas for reading the csv
import pandas as pd


# =====================
# CODE
# =====================

# a function to load the data given by filepath
def load_data(filepath):
    # load the data with pd
    data = pd.read_csv(filepath)
    # put data into np array
    data = np.array(data)

    # get the dimensions (m rows: number of training examples, n columns: example label + pixel values)
    m, n = data.shape

    # shuffle the data to reduce the chance of over-fitting
    random.shuffle(data)

    # transpose the data so each column is an example and not each row
    data = data.T
    # first element will always be the label for the example
    setY = data[0]
    # the rest of the data will be all of the examples
    setX = data[1:]

    # finally, return the x and y data
    return setX, setY

# ----------------------------------------------

# a function to initialise the parameters for the NN
# PARAMS: 
def initialise_parameters():
    # note a variable between the layers x and y is labeled as <var>xy
    # weights
    w12 = np.random.rand(10, 784)   - 0.5
    w23 = np.random.rand(10, 10)    - 0.5

    # biases
    b12 = np.random.rand(10, 1)     - 0.5
    b23 = np.random.rand(10, 1)     - 0.5

    return w12, w23, b12, b23

# -----------------------------------------------

# activation function ReLU (works on a list of variables)
def relu(Z):
    return np.maximum(Z, 0)


# derrivative of ReLU
def derrivative_relu(Z):
    return (Z > 0)

# -----------------------------------------------

# activation function softmax (works on a list of variables)
def softmax(Z):
    return (np.exp(Z) / sum(np.exp(Z)))

# ------------------------------------------------

# the forward propagation function
def forward_propagation(X, w12, w23, b12, b23):
    # get the value of the latent variables
    Z2 = w12.dot(X) + b12
    # activate the hidden layer
    L2 = relu(Z12)
    # get the output neurons
    Z3 = w23.dot(L2) + b23
    # activate the output neurons
    L3 = softmax(Z3)

    # finally return the layers and latent variables
    return Z2, Z3, L2, L3

# ------------------------------------------------

# a function to get the accuracy of prediction (L3) vs actual labels (Y)
def get_accuracy(predictions, Y):
    # return sum of correct predictions / number of labels
    return (np.sum(predictions == Y) / Y.size)


# a function to get the actual prediction from the output layer
def get_prediction(predictions) :
    # its essentially returning the element with the highest number (which corresponds to confidence)
    return np.argmax(predictions, 0)

# ======================
# MAIN
# ======================

def main():
    load_data(r"./archive/mnist_train.csv")

if __name__ == "__main__":
    main()


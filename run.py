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
    setX = data[1:] / 255 # have to divide or the numbers are too big otherwise

    # finally, return the x and y data, realised i also need m for back prop
    return setX, setY, m

# ----------------------------------------------

# a function to initialise the parameters for the NN
# PARAMS: 
def initialise_parameters():
    # note a variable between the layers x and y is labeled as <var>xy
    # weights
    W12 = np.random.rand(10, 784)   - 0.5
    W23 = np.random.rand(10, 10)    - 0.5

    # biases
    b12 = np.random.rand(10, 1)     - 0.5
    b23 = np.random.rand(10, 1)     - 0.5

    return W12, W23, b12, b23

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
def forward_propagation(X, W12, W23, b12, b23):
    # get the value of the latent variables
    Z2 = W12.dot(X) + b12
    # activate the hidden layer
    L2 = relu(Z2)
    # get the output neurons
    Z3 = W23.dot(L2) + b23
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
def get_predictions(predictions) :
    # its essentially returning the element with the highest number (which corresponds to confidence)
    return np.argmax(predictions, 0)

# ------------------------------------------------

# a function to calculate all of the deltas 
def backward_propagation(X, Y, W12, W23, Z2, Z3, L2, L3, m):
    # FIRST PUT LABEL DATA INTO CORRECT FORMAT
    # first create an all zeros np array with the right shape
    normY = np.zeros((Y.size, Y.max() + 1))
    # put in the values into the new array
    normY[np.arange(Y.size), Y] = 1
    # finally just transpose the array
    normY = normY.T

    # CALCULATE THE VARIABLES (ctrl + f comment in README to find the equation)
    # \delta Z_o
    dZ3     = L3 - normY
    # \delta W_o
    dW23    = (dZ3.dot(L2.T)) / m
    # \delta B_o
    db23    = (np.sum(dZ3)) / m
    # \delta Z_h
    dZ2     = W23.T.dot(dZ3) * derrivative_relu(Z2)
    # \delta W_h
    dW12     = (dZ2.dot(X.T)) / m
    # \delta B_h
    db12    = (np.sum(dZ2)) / m

    # finally, return the values
    return db12, db23, dW12, dW23

# ------------------------------------------------

# a function to update the parameters given the weghts, deltas and the learning rate
def update_parameters(b12, b23, W12, W23, db12, db23, dW12, dW23, m, alpha):
    # INPUT LAYER 
    # update the weights between first and hidden layers
    W12     = W12 - (alpha * dW12)
    # update the bias between first and hidden layers
    b12     = b12 - (alpha * db12)

    # HIDDEN LAYER
    # update the weights between hidden and output layers
    W23     = W23 - (alpha * dW23)
    # update the bias between the hiddne and output layers
    b23     = b23 - (alpha * db23)

    # return the updated parameters
    return b12, b23, W12, W23

# ------------------------------------------------

# performs one step of the gradient descent function: forward -> backward -> update params
def gradient_descent(X, Y, W12, W23, b12, b23, m, alpha, generations):
    # loop through generations
    for i in range(generations):
        # forward propagation
        Z2, Z3, L2, L3 = forward_propagation(X, W12, W23, b12, b23)
        # backward propagation
        db12, db23, dW12, dW23 = backward_propagation(X, Y, W12, W23, Z2, Z3, L2, L3, m)
        # update the parameters
        b12, b23, W12, W23 = update_parameters(b12, b23, W12, W23, db12, db23, dW12, dW23, m, alpha)

        # every 99th generation print the accuracy (otherwise it skips out the last 100 in the print statement)
        if i % 99 == 0:
            # get the actual predictions
            predictions = get_predictions(L3)
            print(f"Iteration: {i}\nAccuracy: {get_accuracy(predictions, Y):.2f}\n\t-----------")

    # finally, return the parameters
    return W12, W23, b12, b23


# ======================
# MAIN
# ======================

def main():
    # load training data
    X, Y, m = load_data(r"./archive/mnist_train.csv")
    # initialise parameters
    W12, W23, b12, b23 = initialise_parameters()

    # set the constants
    alpha = 0.05
    generations = 1000

    # perform gradient descent
    W12, W23, b12, b23 = gradient_descent(X, Y, W12, W23, b12, b23, m, alpha, generations)

# ------------------------------------------------

# don't run if imported 
if __name__ == "__main__":
    main() 


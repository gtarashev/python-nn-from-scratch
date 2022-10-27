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

# a function to load the data and split it up 
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
    trainY = data[0]
    # the rest of the data will be all of the examples
    trainX = data[1:]

# ======================
# MAIN
# ======================

def main():
    load_data(r"./archive/mnist_train.csv")

if __name__ == "__main__":
    main()


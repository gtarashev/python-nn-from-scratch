# Python Neural Network only using NumPy
A Neural Network written in Python without the use of any ML libraries, so only NumPy. It will be trained on the [MNIST](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download) dataset, but in csv format, because it is much easier to use.

## Network Structure
#### Syntax (note the layer is the list of activated latent variables):
 * $L_{in}$: the input layer
 * $L_{out}$: the output layer
 * $L_h$: the hidden layer (there will be only one)
 * $\sigma_x$: activarion function where $x$ signifies the actual function itself
 * $z$: the latent variables
 * $Z_l$: list of all variables in layer $l$
 * $x$: input neurons
 * $y$: output neurons
 * ($b_1, ..., b_n$): the weights, $n$ is the number of neurons
 * $b_0$: the bias

#### Predicting (Forward Propagation)
To get a prediction out of the network, I will be using forward propagation, specifically with ReLU as my activation function for my hidden layer and softmax for my output layer. Here's the maths:
$Z_h = b_0 + \sum ^m _i=1 (x _i + b_i )$

## Requirements 
The requirements for the project are:
* numpy 
* pandas

They can be installed using `pip install -r requirements.txt`

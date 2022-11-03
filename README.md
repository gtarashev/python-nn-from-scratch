# Python Neural Network only using NumPy
A Neural Network written in Python without the use of any ML libraries, so only NumPy. It will be trained on the [MNIST](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download) dataset, but in csv format, because it is much easier to use.

## Requirements 
The requirements for the project are:
* numpy 
* pandas

They can be installed using `pip install -r requirements.txt`

## Running
To run the program simply run `python3 run.py`

## Network Structure
__NOTE: GitHub uses latex syntax for inline maths, the syntax is correct (tested in latex) but my browser is not displaying some of it correctly__

### Syntax (note the layer is the list of activated latent variables):
 * $X$: The inputs
 * $Y$: The outputs
 * $L_{i}$: the input layer
 * $L_{o}$: the output layer
 * $L_h$: the hidden layer (there will be only one)
 * $\sigma_x$: activation function where $x$ signifies the actual function itself
 * $z$: the latent variables
 * $Z_l$: list of non-activated variables in layer $l$
 * $x$: input neurons
 * $y$: output neurons
 * $(b_1, ..., b_n)$: the weights, $n$ is the number of neurons, or $W_l$
 * $b_0$: the bias, or $b_l$ when talking about the bias of a specific layer

#### Predicting (Forward Propagation)
To get a prediction out of the network, I will be using forward propagation, with ReLU for the hidden layer and softmax for the output layer. I will use ReLU because it acts like a linear function but it isn't, meaning that complex features can be learned and I can also use gradient descent for my backpropagation. I will be using softmax for the output layer because it is great for multi-class classification problems, which is exactly what I am doing here, another bonus is that softmax will give me a confidence (out of 1) for the given prediction. Here is the maths for forward propagation, starting with the equation for the latent variables:

$z = b_0 + \sum ^{m}_{i=1} (x_i * b_i )$

and $Z_h = \{z_1, z_2, ..., z_n\}$, also:
$L_h = g_{ReLU}(Z_h)$.

Then, to get the output variables we can use:

$y=b_0 + \sum ^m_{i=1} (g_{ReLU}(z_i) * b_i)$

where $Z_{o} = \{y_1, y_2, ..., y_n\}$, and finally activate the neurons to get: $L_{o} = g_{softmax}(Z_{o})$


#### Learning (Backward Propagation + Parameter Updating)
To enable the neural network to learn, I will use gradient descend to adjust the weights, this is one of the reasons that I picked ReLU for my activation function. The main function is:

$\theta _j \leftarrow \theta _ j - \alpha \frac{\delta C}{\delta \theta _j}$

where $\theta_j$ is the $j$-th parameter and $C$ is the cost or error and $\alpha$ is the learning rate. The actual adapted equations are:

$\delta Z_o = A_o - Y$

$\delta W_o = \frac{\delta Z_o A_h ^\intercal}{m}$

$\delta B_o = \frac{\sum \delta Z_o}{m}$

$\delta Z_h = W_h ^\intercal \delta Z_h \cdot g' _{ReLU}(z_h)$

$\delta W_h = \frac{\delta Z_h A_i ^\intercal}{m}$

$\delta B_h = \frac{\sum \delta Z_h}{m}$

and then the parameters are updated with the following equations:

$W_o \leftarrow W_o - \alpha \delta W_o$

$B_o \leftarrow B_o - \alpha \delta B_o$

$W_h \leftarrow W_h - \alpha \delta W_h$

$B_h \leftarrow B_h - \alpha \delta B_h$


## Sample output
I have completed a run, with $\alpha = 0.05$, to show what the output should look like (numbers might be different due to difference in parameter initialisation):

![alt text](https://github.com/gtarashev/python-nn-from-scratch/blob/main/sample_output_0.05alpha.png?raw=true)

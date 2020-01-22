# Introduction

The aim of this project is to implement a simple neural net which is trained using the inverse BFGS method as an optimization method for training the model parameters.

# Getting started

To run the project, install all necessary requirements, given in `requirements.txt` in a python environment. The program can then be run with the command  
`> python main.py`

The following commandline options are available:

| option                               | description                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| -h, --help                           | show help message and exit                                                       |
|  --optimizer OPTIMIZER, -o OPTIMIZER | Type of optimizer to use. Options: backprop, ibfgs, armijo (default: backprop)           |
|  --seed SEED, -s SEED                | Random seed (default: 5000)                                                      |
|  --nsteps NSTEPS, -n NSTEPS          | Number of iterations to compute. Default is 10000 for backprop, 3000 for armijo and 100 for ibfgs |
|  --visualize, -v                     | Visualize training progress                                                      |

# Testing

For testing optimization methods the script `test.py` has been provided.  
In this file the loss function is minimized, with respect to the weights of an un-activated linear layer with entries only on the diagonal. Hance, for a model input of `[1,1]` in 2D the model simply returns the weights. This allows for the weights to be used as input for the given loss function (d.g. Rosenbrock) and hence the training is simply an optimization of the loss function with respect to the model weights (= model output for input `[1,1]` in 2D).

# Known issues

* iBFGS now works for Rosenbrock in 2D, but Armijo line search still stagnates for neural net

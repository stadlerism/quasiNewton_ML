# Introduction

The aim of this project is to implement a simple neural net which is trained using the inverse BFGS method as an optimization method for training the model parameters.

# Getting started

To run the project, install all necessary requirements, given in `requirements.txt` in a python environment. The program can then be run with the command  
`> python main.py`

The following commandline options are available:

| option                               | description                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| -h, --help                           | show help message and exit                                                       |
|  --optimizer OPTIMIZER, -o OPTIMIZER | Type of optimizer to use. Options: backprop, ibfgs (default: backprop)           |
|  --seed SEED, -s SEED                | Random seed (default: 5000)                                                      |
|  --nsteps NSTEPS, -n NSTEPS          | Number of iterations to compute. Default is 100000 for backprop and 10 for ibfgs |
|  --visualize, -v                     | Visualize training progress                                                      |

# Known issues

* The Powell-Wolfe linesearch algorithm does not terminate.

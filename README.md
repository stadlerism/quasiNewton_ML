# Introduction

The aim of this project is to implement a simple neural net which is trained using the inverse BFGS method as an optimization method for training the model parameters.

# Getting started

To run the project, install all necessary requirements, given in `requirements.txt` in a python environment. The program can then be run with the command  
`> python main.py`

The following commandline options are available:

| option                               | description                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| -h, --help                           | show help message and exit                                                       |
|  --optimizer OPTIMIZER, -o OPTIMIZER | Type of optimizer to use. Options: <br>* backprop (default) <br>* bfgs <br>* ibfgs <br>* armijo <br>* bbv1 (Barzilai-Borwein v1) <br>* bbv2 (Barzilai-Borwein v2) <br>* bbv3 (Barzilai-Borwein alternating)           |
|  --seed SEED, -s SEED                | Random seed (default: 5000)                                                      |
|  --nsteps NSTEPS, -n NSTEPS          | Number of iterations to compute. Default is 10000 for backprop, 3000 for armijo and 100 for ibfgs |
|  --visualize, -v                     | Visualize training progress                                                      |
|  --batchsize, -b                     | Batch size for training the network (default: 10)

# Testing

For testing optimization methods the script `test.py` has been provided.  
In this file the loss function is minimized, with respect to the weights of an un-activated linear layer with entries only on the diagonal. Hance, for a model input of `[1,1]` in 2D the model simply returns the weights. This allows for the weights to be used as input for the given loss function (d.g. Rosenbrock) and hence the training is simply an optimization of the loss function with respect to the model weights (= model output for input `[1,1]` in 2D).

# "Real world" example

A small scale, real world example has also been implemented  for comparison of runtime. It can be run with the command  
`> python world.py`

Interesting and well working commandline argument arrangements are
```
> python world.py -v -n 16000
> python world.py -v -n 2000 -o bbv2
> python world.py -v -n 300 -o ibfgs
```

Cases where the convergence turns out to be rather slow are
```
> python world.py -v -n 500 -o armijo
> python world.py -v -n 500 -o bfgs
```

# Known issues

* iBFGS now works for Rosenbrock in 2D, but Armijo line search still stagnates for neural net

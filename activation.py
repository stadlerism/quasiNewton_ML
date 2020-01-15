import numpy as np

class Sigmoid:
    def __call__(self, x):
        res = 1/(1+np.exp(-x))
        return res

    def derivative(self, x):
        sx = self(x)
        return sx*(1-sx)

class SoftMax:
    def __call__(self, x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()
import numpy as np

class Sigmoid:
    def __call__(self, x):
        cutoff = -7e2
        feasible_idx = x > cutoff
        res = np.zeros(x.shape)
        res[feasible_idx] = 1/(1+np.exp(-x[feasible_idx]))
        # res = 1/(1+np.exp(-x))
        return res

    def derivative(self, x):
        sx = self(x)
        return sx*(1-sx)

class ReLU:
    def __call__(self, x):
        x[x<0] = 0.0
        return x

    def derivative(self, x):
        return (x>0).astype(np.float)

class Tanh:
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        pass

class SoftMax:
    def __call__(self, x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

class Identity:
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return np.ones(x.shape)
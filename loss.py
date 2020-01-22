import numpy as np

class L2Loss:
    def __call__(self, x, y):
        return ((x-y)**2).sum()/2

    def dx(self, x, y):
        return (x-y)

class Rosenbrock:
    def __call__(self, x, y):
        d = x-y
        return (1-d[0])**2 + 100*(d[1]-d[0]**2)**2

    def dx(self, x, y):
        d = x-y
        dx = np.zeros(x.shape)
        dx[0] = -400*(d[1]-d[0]**2)*d[0] - 2*(1-d[0])
        dx[1] = 200*(d[1]-d[0]**2)
        return dx
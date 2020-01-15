import numpy as np

class L2Loss:
    def __call__(self, x, y):
        return ((x-y)**2).sum()/2

    def dx(self, x, y):
        return (x-y)
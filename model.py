import numpy as np

from layers import LinearLayer, ActivatedLayer
from activation import Sigmoid, SoftMax
from loss import L2Loss

class LinearModel:
    def __init__(self, widths=[2, 2, 3, 2], lr=0.9, loss=L2Loss()):
        sigmoid = Sigmoid()
        self._layers = []

        for n_in, n_out in zip(widths[:-1], widths[1:]):
            linearLayer = LinearLayer(n_in, n_out, bias=True, lr=lr)
            self._layers.append(ActivatedLayer(linearLayer, sigmoid))

        self._loss = loss

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    @property
    def nparams(self):
        nparams = 0
        for layer in self._layers:
            nparams += layer.nparams
        return nparams

    def train_step(self, x, target):
        res = self(x)

        grad = self._loss.dx(res, target)
        for layer in reversed(self._layers):
            grad = layer.backprop(grad)
            layer.apply_grad()

        return res
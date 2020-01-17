import numpy as np

from layers import LinearLayer, ActivatedLayer, LinearLayer1D
from activation import Sigmoid, SoftMax, Identity
from loss import L2Loss

class LinearModel:
    def __init__(self, widths=[2, 2, 3, 2], lr=0.05, loss=L2Loss()):
        sigmoid = Sigmoid()
        self._layers = []
        self._lr = lr

        for n_in, n_out in zip(widths[:-1], widths[1:]):
            linearLayer = LinearLayer(n_in, n_out, bias=True)
            self._layers.append(ActivatedLayer(linearLayer, sigmoid))

        self._loss = loss

    def __call__(self, x, sigma=None, d=None):
        if sigma is None or d is None:
            for layer in self._layers:
                x = layer(x)
        else:
            dirs = self.full_dir_to_parts(d)
            for layer, d in zip(self._layers, dirs):
                x = layer(x, sigma, d[0], d[1])
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
            grad, updates = layer.backprop(grad, get_grad=True)
            layer.apply_update(-self._lr*updates[0], -self._lr*updates[1])

        return res
        
    def train_step_optimizer(self, x, target, optimizer):
        res = self(x)

        def tmp_loss(res):
            return self._loss(res, target)

        def get_grad_tmp(res):
            return self.get_full_grad(res, target)

        full_update = optimizer.step(x, self.__call__, get_grad_tmp, tmp_loss)
        updates = self.full_dir_to_parts(full_update)
        for layer, u in zip(self._layers, updates):
            layer.apply_update(u[0], u[1])

        return res

    def get_full_grad(self, res, target):
        # get vector containing gradients of all parameters

        grad = self._loss.dx(res, target)
        grads = []
        # iterate through layers in reverse order to compute gradients via backpropagation
        for layer in reversed(self._layers):
            grad, updates = layer.backprop(grad, get_grad=True)
            grads.append(updates)

        full_grad = np.zeros((self.nparams, 1))
        idx = 0

        # create single vector of all gradients in correct order
        for wd, bd in reversed(grads):
            dim_w = np.prod(wd.shape)
            full_grad[idx:idx+dim_w] = wd.reshape(-1,1)
            idx += dim_w
            dim_b = np.prod(bd.shape)
            full_grad[idx:idx+dim_b] = bd.reshape(-1,1)
            idx += dim_b
        return full_grad

        
    def full_dir_to_parts(self, full_update):
        updates = []
        idx = 0
        size = 0
        for layer in self._layers:
            shape_w = layer.shape[0]
            shape_b = layer.shape[1]
            size_w = np.prod(shape_w)
            size_b = np.prod(shape_b)
            updates.append((
                np.reshape(full_update[idx:idx+size_w], shape_w),
                np.reshape(full_update[idx+size_w:idx+size_w+size_b], shape_b)
            ))
            idx += size_w + size_b
        return updates

class TestModel(LinearModel):
    # Model to test correctness of algorithms
    # will only return the vector stored in linearLayer multiplied elementwise with the input
    def __init__(self, n, lr=0.05, loss=L2Loss()):
        identity = Identity()
        self._layers = []
        self._lr = lr

        linearLayer = LinearLayer1D(n)
        self._layers.append(ActivatedLayer(linearLayer, identity))

        self._loss = loss
import numpy as np

class LinearLayer:
    def __init__(self, n_in, n_out, bias=True, lr=0.9):
        self._w = np.random.randn(n_out, n_in)
        self._b = np.random.randn(n_out, 1)
        self._lr = lr
        self._activation = None
        self._w_grad = None
        self._b_grad = None

    def __call__(self, x):
        return self.forward(x)

    @property
    def nparams(self):
        return np.prod(self._w.shape) + np.prod(self._b.shape)

    def forward(self, x):
        self._activation = x
        nsamples = 1
        if len(x.shape) == 2:
            nsamples = x.shape[1]
        return np.matmul(self._w, x) + np.repeat(self._b, nsamples, axis=1)


    def backprop(self, grad, get_grad=False):
        new_grad = np.matmul(self._w.transpose(), grad)
        self._w_grad = self._lr * np.matmul(grad, self._activation.transpose())
        self._b_grad = self._lr * np.average(grad, axis=1).reshape(-1,1)
        if not get_grad:
            return new_grad
        else:
            return new_grad, (self._w_grad, self._b_grad)

    def apply_grad(self):
        self.apply_update(-self._w_grad, -self._b_grad)

    def apply_update(self, wd, bd):
        self._w += wd
        self._b += bd



class ActivatedLayer:
    def __init__(self, layer, activation):
        self._layer = layer
        self._activation = activation
        self._dx = None

    def __call__(self, x):
        return self.forward(x)

    @property
    def nparams(self):
        return self._layer.nparams

    @property
    def _w(self):
        return self._layer._w
    
    @property
    def _b(self):
        return self._layer._b

    def forward(self, x):
        x = self._layer(x)
        self._dx = self._activation.derivative(x)
        return self._activation(x)

    def backprop(self, grad, **kwargs):
        grad = np.multiply(self._dx, grad)
        return self._layer.backprop(grad, **kwargs)

    def apply_grad(self):
        self._layer.apply_grad()

    def apply_update(self, wd, bd):
        self._layer.apply_update(wd, bd)

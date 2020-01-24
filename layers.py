import numpy as np

class LinearLayer1D:
    def __init__(self, n, d=None):
        self._n = n
        if d is None:
            self._d = np.ones((n,1))#np.random.randn(n,1)
        else:
            self._d = d
        self._activation = None

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    @property
    def nparams(self):
        return self._n

    def forward(self, x, sigma=None, dd=None, *args, **kwargs):
        self._activation = x
        if sigma is None:
            d_tmp = self._d
        else:
            d_tmp = self._d + sigma*dd
        nsamples = 1
        if len(x.shape) == 2:
            nsamples = x.shape[1]
        return np.repeat(d_tmp, nsamples, axis=1) * x

    def backprop(self, grad, get_grad=False):
        new_grad = self._d * grad
        d_grad = grad * self._activation
        if not get_grad:
            return new_grad
        else:
            return new_grad, (d_grad, np.zeros((0,0)))

    def apply_update(self, dd, *args, **kwargs):
        self._d += dd

    @property
    def shape(self):
        return ((self._n, 1), (0,0))

class LinearLayer:
    def __init__(self, n_in, n_out, bias=True):
        self._w = np.random.randn(n_out, n_in)
        self._b = np.random.randn(n_out, 1)
        self._activation = None

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    @property
    def nparams(self):
        return np.prod(self._w.shape) + np.prod(self._b.shape)

    def forward(self, x, sigma=None, wd=None, bd=None):
        self._activation = x
        if sigma is None:
            w_tmp = self._w
            b_tmp = self._b
        else:
            w_tmp = self._w + sigma*wd
            b_tmp = self._b + sigma*bd
        nsamples = 1
        if len(x.shape) == 2:
            nsamples = x.shape[1]
        return np.matmul(w_tmp, x) + np.repeat(b_tmp, nsamples, axis=1)

    def backprop(self, grad, get_grad=False):
        new_grad = np.matmul(self._w.transpose(), grad)
        w_grad = np.matmul(grad, self._activation.transpose())
        b_grad = np.average(grad, axis=1).reshape(-1,1)
        if not get_grad:
            return new_grad
        else:
            return new_grad, (w_grad, b_grad)

    def apply_update(self, wd, bd):
        self._w += wd
        self._b += bd

    @property
    def shape(self):
        return (self._w.shape, self._b.shape)



class ActivatedLayer:
    def __init__(self, layer, activation):
        self._layer = layer
        self._activation = activation
        self._dx = None

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    @property
    def nparams(self):
        return self._layer.nparams

    def forward(self, x, *args, **kwargs):
        x = self._layer.forward(x, *args, **kwargs)
        self._dx = self._activation.derivative(x)
        return self._activation(x)

    def backprop(self, grad, *args, **kwargs):
        grad = np.multiply(self._dx, grad)
        return self._layer.backprop(grad, *args, **kwargs)

    def apply_update(self, *args, **kwargs):
        self._layer.apply_update(*args, **kwargs)

    @property
    def shape(self):
        return self._layer.shape


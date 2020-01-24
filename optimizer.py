import numpy as np

class IterationCompleteException(Exception):
    pass


# TODO: non-monotonous line search

class BaseOptimizer:
    def __init__(self, nparams, *args, beta=0.5, gamma=0.0001, **kwargs):
        self._nparams = nparams
        self._beta = beta
        self._gamma = gamma

    def ArmijoSearch(self, x, lfx, gfx, d, f, loss):
        sigma = np.float(1.0)
        fx_step = f(x, sigma, d)
        lfx_step = loss(fx_step)
        # --> decrease sigma_min until we satisfy PW1
        while not self.PW1(lfx, lfx_step, gfx, d, sigma):
            sigma *= self._beta
            fx_step = f(x, sigma, d)
            lfx_step = loss(fx_step)
        return sigma

    def PW1(self, lfx, lfx_step, gfx, d, sigma):
        return (lfx_step - lfx <= sigma * self._gamma * np.dot(gfx.flatten(), d.flatten()))

    def PW2(self, gfx, gfx_step, d):
        return (np.dot(gfx_step.flatten(), d.flatten()) >= self._eta * np.dot(gfx.flatten(), d.flatten()))

    def step(self, x, f, gf, loss):
        raise NotImplementedError()

    def get_dir(self, gfx):
        raise NotImplementedError()

    def linesearch(self, x, f, gf, d, lfx, gfx, loss):
        raise NotImplementedError()


class DescentMethod(BaseOptimizer):
    def __init__(self, nparams, *args, **kwargs):
        super(DescentMethod, self).__init__(nparams, *args, **kwargs)
        self._s = None

    def get_dir(self, gfx):
        if np.linalg.norm(gfx) < 5e-15:
            # print(np.linalg.norm(gfx))
            raise  IterationCompleteException()
        return -gfx

    def step(self, x, f, gf, loss):
        fx = f(x)
        gfx = gf(fx)
        lfx = loss(fx)
        d = self.get_dir(gfx)
        sigma = self.linesearch(x, f, gf, d, lfx, gfx, loss)
        self._s = sigma*d
        return sigma*d


class QuasiNewton(BaseOptimizer):
    def __init__(self, nparams, *args, M_0 = None, **kwargs):
        super(QuasiNewton, self).__init__(nparams, *args, **kwargs)
        if M_0 is None:
            self._matrix = np.identity(nparams)
        else:
            self._matrix = M_0

        self._s = None
        self._prev_gfx = None

    def step(self, x, f, gf, loss):
        fx = f(x)
        gfx = gf(fx)
        lfx = loss(fx)
        self.update(gfx)
        d = self.get_dir(gfx)
        sigma = self.linesearch(x, f, gf, d, lfx, gfx, loss)
        self._s = sigma*d
        return sigma*d

    def get_dir(self, gfx):
        if np.linalg.norm(gfx) < 5e-15:
            raise  IterationCompleteException()
        d = -np.dot(self._matrix, gfx)
        if np.dot(d.flatten(), gfx.flatten()) >= 0:
            d = -gfx
        return d

    def update(self, gfx):
        raise NotImplementedError()

    def linesearch(self, x, f, gf, d, lfx, gfx, loss):
        raise NotImplementedError()



class SteepDescent(DescentMethod):
    def __init__(self, nparams, *args, beta=1/2, gamma=0.4, **kwargs):
        super(SteepDescent, self).__init__(nparams, *args, beta=beta, gamma=gamma, **kwargs)

    def linesearch(self, x, f, gf, d, lfx, gfx, loss):
        return self.ArmijoSearch(x, lfx, gfx, d, f, loss)


class BarzilaiBorwein(DescentMethod):
    def __init__(self, nparams, *args, strategy='v2', **kwargs):
        super(BarzilaiBorwein, self).__init__(nparams, *args, **kwargs)
        self._i = 0
        strategy = strategy.lower()
        if strategy == 'alternate':
            strategy = 'alt'
        self._strategy = strategy
        self._prev_gfx = 0
        self._prev_step = None

    def linesearch(self, x, f, gf, d, lfx, gfx, loss):
        y = gfx - self._prev_gfx
        if self._prev_step is None:
            s = d
        else:
            s = self._prev_step
        
        if self._strategy == 'v1' or (self._strategy == 'alt' and self._i % 2 == 0):
            sTs = np.dot(s.flatten(), s.flatten())
            if np.abs(sTs) < 1e-50:
                raise IterationCompleteException()
            sTy = np.dot(s.flatten(), y.flatten())
            alpha = sTy / sTs
        else:
        # elif self._strategy == 'v2' or (self._strategy == 'alt' and self._i % 2 == 1)
            sTy = np.dot(s.flatten(), y.flatten())
            if np.abs(sTy) < 1e-50:
                raise IterationCompleteException()
            yTy = np.dot(y.flatten(), y.flatten())
            alpha = yTy / sTy
        
        if alpha > 0:
            sigma = 1/alpha
        else:
            sigma = self.ArmijoSearch(x, lfx, gfx, d, f, loss)

        self._i += 1
        self._prev_gfx = gfx
        self._prev_step = sigma*d
        return sigma


class InverseBFGS(QuasiNewton):
    def __init__(self, nparams, *args, gamma=0.4, eta=0.9, M_0 = None, **kwargs):
        super(InverseBFGS, self).__init__(nparams, *args, M_0=M_0, **kwargs)
        self._gamma = gamma
        self._eta = eta

    def update(self, gfx):
        # Inverse BFGS update
        # only done once a step has been completed
        if not self._prev_gfx is None:
            if self._s is None:
                raise Exception("Implementation error: No step has been completed between calls of update.")
            else:
                s = self._s
                y = gfx - self._prev_gfx
                z = s - np.dot(self._matrix, y)
                sTy = np.dot(s.flatten(), y.flatten())
                if np.abs(sTy) < 1e-100:
                    raise  IterationCompleteException()

                zsT = np.dot(z, s.transpose())
                zTy = np.dot(z.flatten(), y.flatten())
                ssT = np.dot(s, s.transpose())
                self._matrix += (zsT+zsT.transpose())/sTy - zTy/(sTy**2)*ssT
        
        self._d = None
        self._prev_gfx = gfx

    def linesearch(self, x, f, gf, d, lfx, gfx, loss):
        # calculate step length using Powell-Wolfe criteria
        # Note: f is to be optimized; x is just the input to f and not to be optimized!
        sigma = np.float(1.0)
        fx_step = f(x, sigma, d)
        lfx_step = loss(fx_step)
        if self.PW1(lfx, lfx_step, gfx, d, sigma):
            gfx_step = gf(fx_step)
            if self.PW2(gfx, gfx_step, d):
                # we already got a solution! --> return
                return sigma

            # solution to PW1, but not to PW2
            # --> increase sigma_max until we no longer satisfy PW1
            sigma_max = np.float(2.0)
            fx_step = f(x, sigma_max, d)
            lfx_step = loss(fx_step)
            while self.PW1(lfx, lfx_step, gfx, d, sigma_max):
                sigma_max *= 2
                fx_step = f(x, sigma_max, d)
                lfx_step = loss(fx_step)
            sigma_min = sigma_max / 2
        else:
            # no solution to PW1
            # --> decrease sigma_min until we satisfy PW1
            sigma_min = np.float(1/2)
            fx_step = f(x, sigma_min, d)
            lfx_step = loss(fx_step)
            while not self.PW1(lfx, lfx_step, gfx, d, sigma_min):
                sigma_min /= 2
                fx_step = f(x, sigma_min, d)
                lfx_step = loss(fx_step)
            sigma_max = sigma_min * 2

        # we got an interval [sigma_min, sigma_max] containing a solution
        # --> shrink interval until we find one
        fx_step = f(x, sigma_min, d)
        lfx_step = loss(fx_step)
        gfx_step = gf(fx_step)
        while not self.PW2(gfx, gfx_step, d):
            sigma = (sigma_min + sigma_max) / 2
            fx_step = f(x, sigma, d)
            lfx_step = loss(fx_step)
            if self.PW1(lfx, lfx_step, gfx, d, sigma):
                sigma_min = sigma
                gfx_step = gf(fx_step)
            else:
                sigma_max = sigma

        # sigma_min satisfies both PW1 and PW2
        return sigma_min
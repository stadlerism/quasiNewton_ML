import numpy as np

from lineSearch import ArmijoSearch, PowellWolfeSearch, BarzilaiBorweinSearch
from exceptions import IterationCompleteException


# TODO: non-monotonous line search

class BaseOptimizer:
    def __init__(self, nparams, *args, **kwargs):
        self._nparams = nparams

    def step(self, *args, **kwargs):
        raise NotImplementedError()

    def get_dir(self, *args, **kwargs):
        raise NotImplementedError()


class DescentMethod(BaseOptimizer):
    def __init__(self, nparams, *args, linesearch=ArmijoSearch, **kwargs):
        super(DescentMethod, self).__init__(nparams, *args, **kwargs)
        self._linesearch = linesearch
        self._s = None
        self._line_search_kwargs = kwargs

    def get_dir(self, gfx):
        if np.linalg.norm(gfx) < 1e-12:
            # print(np.linalg.norm(gfx))
            raise  IterationCompleteException()
        return -gfx

    def linesearch(self, *args, **kwargs):
        return self._linesearch(*args, **kwargs)

    def step(self, x, f, gf, loss):
        fx = f(x)
        gfx = gf(fx)
        lfx = loss(fx)
        d = self.get_dir(gfx)
        sigma = self.linesearch(x, f, gf, d, lfx, gfx, loss, **self._line_search_kwargs)
        self._s = sigma*d
        return sigma*d


class NewtonType(BaseOptimizer):
    def __init__(self, nparams, *args, M_0 = None, **kwargs):
        super(NewtonType, self).__init__(nparams, *args, **kwargs)
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
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def linesearch(self, *args, **kwargs):
        raise NotImplementedError()


class QuasiNewton(NewtonType):
    def get_dir(self, gfx):
        if np.linalg.norm(gfx) < 1e-12:
            raise  IterationCompleteException()
        d = -np.linalg.solve(self._matrix, gfx)
        return d


class InverseQuasiNewton(NewtonType):
    def get_dir(self, gfx):
        if np.linalg.norm(gfx) < 1e-12:
            raise  IterationCompleteException()
        d = -np.dot(self._matrix, gfx)
        return d


class BarzilaiBorwein(DescentMethod):
    def __init__(self, nparams, *args, strategy='v2', **kwargs):
        super(BarzilaiBorwein, self).__init__(nparams, *args, **kwargs)
        strategy = strategy.lower()
        if strategy == 'alternate':
            strategy = 'alt'
        self._strategy = strategy
        self._prev_gfx = 0
        self._prev_step = None
        self._i = 0

    def linesearch(self, x, f, gf, d, lfx, gfx, loss, **kwargs):
        if self._strategy == 'v1' or (self._strategy == 'alt' and self._i % 2 == 0):
            strategy = 0
        else:
            strategy = 1
        
        if self._prev_step is None:
            self._prev_step = d

        sigma = BarzilaiBorweinSearch(x, f, gf, d, lfx, gfx, loss, self._prev_step, self._prev_gfx, strategy=strategy)
        self._prev_gfx = gfx
        self._prev_step = sigma*d
        self._i += 1

        return sigma

class InverseBFGS(InverseQuasiNewton):
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
                if np.abs(sTy) < 1e-12:
                    self._matrix = np.identity(self._nparams)
                    return

                zsT = np.dot(z, s.transpose())
                zTy = np.dot(z.flatten(), y.flatten())
                ssT = np.dot(s, s.transpose())
                self._matrix += (zsT+zsT.transpose())/sTy - zTy/(sTy**2)*ssT
        
        self._d = None
        self._prev_gfx = gfx

    def linesearch(self, x, f, gf, d, lfx, gfx, loss):
        return PowellWolfeSearch(x, f, gf, d, lfx, gfx, loss, gamma=self._gamma, eta=self._eta)



class BFGS(QuasiNewton):
    def __init__(self, nparams, *args, gamma=0.4, eta=0.9, M_0 = None, **kwargs):
        super(BFGS, self).__init__(nparams, *args, M_0=M_0, **kwargs)
        self._gamma = gamma
        self._eta = eta

    def update(self, gfx):
        # Inverse BFGS update
        # only done once a step has been completed
        # return
        if not self._prev_gfx is None:
            if self._s is None:
                raise Exception("Implementation error: No step has been completed between calls of update.")
            else:
                s = self._s
                y = gfx - self._prev_gfx
                Hs = np.dot(self._matrix, s)
                sTy = np.dot(s.flatten(), y.flatten())
                sTHs = np.dot(s.flatten(), Hs.flatten())
                if np.abs(sTy) < 1e-12 or np.abs(sTHs) < 1e-12:
                    self._matrix = np.identity(self._nparams)
                    return

                HssTHT = np.dot(Hs, Hs.transpose())
                yyT = np.dot(y, y.transpose())
                self._matrix += yyT / sTy - HssTHT / sTHs

        
        self._d = None
        self._prev_gfx = gfx

    def linesearch(self, x, f, gf, d, lfx, gfx, loss):
        return PowellWolfeSearch(x, f, gf, d, lfx, gfx, loss, gamma=self._gamma, eta=self._eta)
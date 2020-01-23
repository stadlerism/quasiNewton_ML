import numpy as np

class IterationCompleteException(Exception):
    pass

class InverseBFGS:
    def __init__(self, nparams, gamma=0.4, eta=0.9, M_0 = None):
        self._nparams = nparams
        self._gamma = gamma
        self._eta = eta
        if M_0 is None:
            self._matrix = np.identity(nparams)
        else:
            self._matrix = M_0

        self._d = None
        self._prev_gfx = None

    def step(self, x, f, gf, loss):
        fx = f(x)
        gfx = gf(fx)
        lfx = loss(fx)
        self.update(gfx)
        s = self.get_dir(gfx)
        sigma = self.linesearch(x, f, gf, s, lfx, gfx, loss)
        self._d = sigma*s
        return sigma*s

    def get_dir(self, gfx):
        if np.linalg.norm(gfx) < 5e-15:
            raise  IterationCompleteException()
        s = -np.dot(self._matrix, gfx)
        if np.dot(s.flatten(), gfx.flatten()) >= 0:
            s = -gfx
        return s

    def update(self, gfx):
        # Inverse BFGS update
        # only done once a step has been completed
        if not self._prev_gfx is None:
            if self._d is None:
                raise Exception("Implementation error: No step has been completed between calls of update.")
            else:
                d = self._d
                y = gfx - self._prev_gfx
                z = d - np.dot(self._matrix, y)
                dTy = np.dot(d.flatten(), y.flatten())
                if np.abs(dTy) < 1e-100:
                    raise  IterationCompleteException()

                zdT = np.dot(z, d.transpose())
                zTy = np.dot(z.flatten(), y.flatten())
                ddT = np.dot(d, d.transpose())
                self._matrix += (zdT+zdT.transpose())/dTy - zTy/(dTy**2)*ddT
        
        self._d = None
        self._prev_gfx = gfx

    def PW1(self, lfx, lfx_step, gfx, s, sigma):
        return (lfx_step - lfx <= sigma * self._gamma * np.dot(gfx.flatten(), s.flatten()))

    def PW2(self, gfx, gfx_step, s):
        return (np.dot(gfx_step.flatten(), s.flatten()) >= self._eta * np.dot(gfx.flatten(), s.flatten()))

    def linesearch(self, x, f, gf, s, lfx, gfx, loss):
        # calculate step length using Powell-Wolfe criteria
        # Note: f is to be optimized; x is just the input to f and not to be optimized!
        sigma = np.float(1.0)
        fx_step = f(x, sigma, s)
        lfx_step = loss(fx_step)
        if self.PW1(lfx, lfx_step, gfx, s, sigma):
            gfx_step = gf(fx_step)
            if self.PW2(gfx, gfx_step, s):
                # we already got a solution! --> return
                return sigma

            # solution to PW1, but not to PW2
            # --> increase sigma_max until we no longer satisfy PW1
            sigma_max = np.float(2.0)
            fx_step = f(x, sigma_max, s)
            lfx_step = loss(fx_step)
            while self.PW1(lfx, lfx_step, gfx, s, sigma_max):
                sigma_max *= 2
                fx_step = f(x, sigma_max, s)
                lfx_step = loss(fx_step)
            sigma_min = sigma_max / 2
        else:
            # no solution to PW1
            # --> decrease sigma_min until we satisfy PW1
            sigma_min = np.float(1/2)
            fx_step = f(x, sigma_min, s)
            lfx_step = loss(fx_step)
            while not self.PW1(lfx, lfx_step, gfx, s, sigma_min):
                sigma_min /= 2
                fx_step = f(x, sigma_min, s)
                lfx_step = loss(fx_step)
            sigma_max = sigma_min * 2

        # we got an interval [sigma_min, sigma_max] containing a solution
        # --> shrink interval until we find one
        fx_step = f(x, sigma_min, s)
        lfx_step = loss(fx_step)
        gfx_step = gf(fx_step)
        while not self.PW2(gfx, gfx_step, s):
            sigma = (sigma_min + sigma_max) / 2
            fx_step = f(x, sigma, s)
            lfx_step = loss(fx_step)
            if self.PW1(lfx, lfx_step, gfx, s, sigma):
                sigma_min = sigma
                gfx_step = gf(fx_step)
            else:
                sigma_max = sigma

        # sigma_min satisfies both PW1 and PW2
        return sigma_min

            
class SteepDescent(InverseBFGS):
    def __init__(self, nparams, beta=1/2, gamma=0.4):
        M_0 = np.zeros((0))
        eta = 0
        self._beta = beta
        super(SteepDescent, self).__init__(nparams, gamma, eta, M_0)

    def step(self, x, f, gf, loss):
        fx = f(x)
        gfx = gf(fx)
        lfx = loss(fx)
        s = self.get_dir(gfx)
        sigma = self.linesearch(x, f, gf, s, lfx, gfx, loss)
        self._d = sigma*s
        return sigma*s

    def linesearch(self, x, f, gf, s, lfx, gfx, loss):
        # calculate step length using Powell-Wolfe criteria
        # Note: f is to be optimized; x is just the input to f and not to be optimized!
        sigma = np.float(1.0)
        fx_step = f(x, sigma, s)
        lfx_step = loss(fx_step)
        # --> decrease sigma_min until we satisfy PW1
        while not self.PW1(lfx, lfx_step, gfx, s, sigma):
            sigma *= self._beta
            fx_step = f(x, sigma, s)
            lfx_step = loss(fx_step)

        return sigma

    def get_dir(self, gfx):
        if np.linalg.norm(gfx) < 5e-15:
            # print(np.linalg.norm(gfx))
            raise  IterationCompleteException()
        return -gfx
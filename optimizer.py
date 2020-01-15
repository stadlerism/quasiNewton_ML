import numpy as np


class InverseBFGS:
    def __init__(self, nparams, gamma=1/2, eta=0.9, M_0 = None):
        self._nparams = nparams
        self._gamma = gamma
        self._eta = eta
        if M_0 is None:
            self._matrix = np.identity(nparams)
        else:
            self._matrix = M_0

        self._d = None
        self._prev_gfx = None

    def step(self, x, f, gf):
        fx = f(x)
        gfx = gf(fx)
        self.update(gfx)
        s = self.get_dir(gfx)
        sigma = self.linesearch(x, f, gf, s, fx, gfx)
        self._d = sigma*s
        return sigma*s

    def get_dir(self, gfx):
        return -np.dot(self._matrix, gfx)

    def update(self, gfx):
        # Inverse BFGS update
        # only done once a step has been completed
        if not self._prev_gfx is None:
            if self._d is None:
                raise Exception("Implementation error: No step has been completed between calls of update.")
            else:
                d = self._d
                y = self._prev_gfx - gfx
                z = d - np.dot(self._matrix, y)
                zdT = np.dot(z, d.transpose())
                dTy = np.dot(y.transpose(), d)
                zTy = np.dot(z.transpose(), y)
                ddT = np.dot(d, d.transpose())
                self._matrix += (zdT + zdT.transpose())/dTy - zTy/(dTy**2)*ddT
        
        self._d = None
        self._prev_gfx = gfx

    def PW1(self, fx, fx_step, gfx, s, sigma):
        return (fx_step - fx <= sigma * self._gamma * np.dot(gfx.flatten(), s.flatten())).all()

    def PW2(self, gfx, gfx_step, s):
        return (np.dot(gfx_step.flatten(), s.flatten()) >= self._eta * np.dot(gfx.flatten(), s.flatten()))

    def linesearch(self, x, f, gf, s, fx, gfx):
        # calculate step length using Powell-Wolfe criteria
        # Note: f is to be optimized; x is just the input to f and not to be optimized!
        sigma = 1
        fx_step = f(x, sigma, s)
        if self.PW1(fx, fx_step, gfx, s, sigma):
            gfx_step = gf(fx_step)
            if self.PW2(gfx, gfx_step, s):
                # we already got a solution! --> return
                print("done")
                return sigma

            print("step 1")
            # solution to PW1, but not to PW2
            # --> increase sigma_max until we no longer satisfy PW1
            sigma_max = 2
            fx_step = f(x, sigma_max, s)
            if self.PW1(fx, fx_step, gfx, s, sigma_max):
                sigma_max *= 2
                fx_step = f(x, sigma_max, s)
            sigma_min = sigma_max / 2
        else:
            print("step 2")
            # no solution to PW1
            # --> decrease sigma_min until we satisfy PW1
            sigma_min = 1/2
            fx_step = f(x, sigma_min, s)
            while not self.PW1(fx, fx_step, gfx, s, sigma_min):
                sigma_min /= 2
                fx_step = f(x, sigma_min, s)
            sigma_max = sigma_min * 2

        print("step 3")
        # we got an interval [sigma_min, sigma_max] containing a solution
        # --> shrink interval until we find one
        fx_step = f(x, sigma_min, s)
        gfx_step = gf(fx_step)
        while not self.PW2(gfx, gfx_step, s):
            sigma = (sigma_min + sigma_max) / 2
            fx_step = f(x, sigma, s)
            if self.PW1(fx, fx_step, gfx, s, sigma_min):
                sigma_min = sigma
                gfx_step = gf(fx_step)
            else:
                sigma_max = sigma

        print("done")
        # sigma_min satisfies both PW1 and PW2
        return self._sigma

            
    
import numpy as np

from exceptions import IterationCompleteException

def PW1(lfx, lfx_step, gfx, d, sigma, gamma):
    return (lfx_step - lfx <= sigma * gamma * np.dot(gfx.flatten(), d.flatten()))

def PW2(gfx, gfx_step, d, eta):
    return (np.dot(gfx_step.flatten(), d.flatten()) >= eta * np.dot(gfx.flatten(), d.flatten()))

def ArmijoSearch(x, f, gf, d, lfx, gfx, loss, beta=0.5, gamma=0.0001, sigma_0 = np.float(1.0)):
    sigma = sigma_0
    fx_step = f(x, sigma, d)
    lfx_step = loss(fx_step)
    # --> decrease sigma_min until we satisfy PW1
    while not PW1(lfx, lfx_step, gfx, d, sigma, gamma):
        sigma *= beta
        fx_step = f(x, sigma, d)
        lfx_step = loss(fx_step)
    return sigma

def PowellWolfeSearch(x, f, gf, d, lfx, gfx, loss, gamma=0.0001, eta=0.9):
    # calculate step length using Powell-Wolfe criteria
    # Note: f is to be optimized; x is just the input to f and not to be optimized!
    sigma = np.float(1.0)
    fx_step = f(x, sigma, d)
    lfx_step = loss(fx_step)
    if PW1(lfx, lfx_step, gfx, d, sigma, gamma):
        gfx_step = gf(fx_step)
        if PW2(gfx, gfx_step, d, eta):
            # we already got a solution! --> return
            return sigma

        # solution to PW1, but not to PW2
        # --> increase sigma_max until we no longer satisfy PW1
        sigma_max = np.float(2.0)
        fx_step = f(x, sigma_max, d)
        lfx_step = loss(fx_step)
        while PW1(lfx, lfx_step, gfx, d, sigma_max, gamma):
            sigma_max *= 2
            fx_step = f(x, sigma_max, d)
            lfx_step = loss(fx_step)
        sigma_min = sigma_max / 2
    else:
        # no solution to PW1
        # --> decrease sigma_min until we satisfy PW1
        sigma_min = ArmijoSearch(x, f, gf, d, lfx, gfx, loss, beta=0.5, gamma=gamma, sigma_0=np.float(0.5))
        sigma_max = sigma_min * 2

    # we got an interval [sigma_min, sigma_max] containing a solution
    # --> shrink interval until we find one
    fx_step = f(x, sigma_min, d)
    lfx_step = loss(fx_step)
    gfx_step = gf(fx_step)
    while not PW2(gfx, gfx_step, d, eta) and (sigma_max-sigma_min) > 1e-10:
        sigma = (sigma_min + sigma_max) / 2
        fx_step = f(x, sigma, d)
        lfx_step = loss(fx_step)
        if PW1(lfx, lfx_step, gfx, d, sigma, gamma):
            sigma_min = sigma
            gfx_step = gf(fx_step)
        else:
            sigma_max = sigma

    # sigma_min satisfies both PW1 and PW2
    return sigma_min

    
def BarzilaiBorweinSearch(x, f, gf, d, lfx, gfx, loss, step_prev, gfx_prev, strategy=0, beta=0.5, gamma=0.0001):
    y = gfx - gfx_prev
    s = step_prev

    alpha = -1
    if strategy == 0:
        sTs = np.dot(s.flatten(), s.flatten())
        if np.abs(sTs) > 1e-30:
            sTy = np.dot(s.flatten(), y.flatten())
            alpha = sTy / sTs
    else:
        sTy = np.dot(s.flatten(), y.flatten())
        if np.abs(sTy) > 1e-30:
            yTy = np.dot(y.flatten(), y.flatten())
            alpha = yTy / sTy
    
    if alpha > 0:
        sigma = 1/alpha
    else:
        sigma = ArmijoSearch(x, f, gf, d, lfx, gfx, loss, beta=0.5, gamma=0.0001)

    return sigma


import numpy as np

class L2Loss:
    def __call__(self, x, y):
        return ((x-y)**2).sum()/2

    def dx(self, x, y):
        return (x-y)

class Rosenbrock:
    # min at (1,1)
    def __call__(self, x, y):
        d = x-y
        return (1-d[0,:])**2 + 100*(d[1,:]-d[0,:]**2)**2

    def dx(self, x, y):
        d = x-y
        dx = np.zeros(x.shape)
        dx[0,:] = -400*(d[1,:]-d[0,:]**2)*d[0,:] - 2*(1-d[0,:])
        dx[1,:] = 200*(d[1,:]-d[0,:]**2)
        return dx

class Beale:
    def __call__(self, x, y):
        d = x-y
        return (1.5- d[0,:] + d[0,:]*d[1,:])**2 + (2.25 - d[0,:] + d[0,:]*d[1,:]**2)**2 + (2.625 - d[0,:] + d[0,:]*d[1,:]**3)**2

    def dx(self, x, y):
        d = x-y
        dx = np.zeros(x.shape)
        dx[0,:] = 2*(1.5- d[0,:] + d[0,:]*d[1,:])*(d[1,:]-1) + 2*(2.25 - d[0,:] + d[0,:]*d[1,:]**2)*(d[1]**2-1) + 2*(2.625 - d[0,:] + d[0,:]*d[1,:]**3)*(d[1,:]**3-1)
        dx[1,:] = 2*(1.5- d[0,:] + d[0,:]*d[1,:])*d[0,:] + 2*(2.25 - d[0,:] + d[0,:]*d[1,:]**2)*(2*d[0,:]*d[1,:]) + 2*(2.625 - d[0,:] + d[0,:]*d[1,:]**3)*(3*d[0,:]*d[1,:]**2)
        return dx
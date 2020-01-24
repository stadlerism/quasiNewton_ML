import numpy as np
import matplotlib.pyplot as plt

def plot_results(model, train_src, continuous=False):
    npoints = 1000
    xi = np.linspace(0,1,npoints)
    x1,x2 = np.meshgrid(xi, xi)
    samples = np.array([
        x1.flatten(),
        x2.flatten()
    ])
    
    results = model(samples)
    if not continuous:
        r = (results[0,:] > results[1,:]).astype(float)
    else:
        r = np.exp(results[0,:])/(np.exp(results[0,:])+np.exp(results[1,:]))
    plt.imshow(r.reshape((npoints, npoints)), extent=(0,1,1,0), interpolation='nearest', cmap=plt.get_cmap('cividis'))
    plt.colorbar()
    plt.plot(train_src[0,5:10], train_src[1,5:10], marker='x', linestyle='')
    plt.plot(train_src[0,0:5], train_src[1,0:5], marker='o', linestyle='')
    plt.show()
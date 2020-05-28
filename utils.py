import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

world_cm = LinearSegmentedColormap.from_list(
    "", [
        (0,(1,1,1,0)),
        (0.6,(1,1,1,0)),
        (0.7,(1,.3,.3,0.8)),
        (1,(1,0,0,1))
    ]
)

def plot_results(model, train_src, train_dst=None, continuous=False, savename=None):
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
    plt.figure()
    plt.imshow(r.reshape((npoints, npoints)), extent=(0,1,1,0), interpolation='nearest', cmap=plt.get_cmap('cividis'))
    plt.colorbar()
    plt.plot(train_src[0,train_dst[0,:]==0], train_src[1,train_dst[0,:]==0], marker='x', linestyle='')
    plt.plot(train_src[0,train_dst[0,:]==1], train_src[1,train_dst[0,:]==1], marker='o', linestyle='')
    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)


def load_world(n_samples=100, p=None):
    full_img = Image.open("images/full.png")
    english_img = Image.open("images/english.png")
    x = np.array(range(0, full_img.width), dtype=np.int)
    y = np.array(range(0, full_img.height), dtype=np.int)
    grid = np.meshgrid(x,y)
    train_grid = np.array([grid[0].flatten()/full_img.width, grid[1].flatten()/full_img.height])

    english_np = np.array(english_img)[:,:,0].flatten()//255
    dst_grid = np.array([english_np, 1-english_np], dtype=np.float)

    full_np = np.array(full_img, dtype=np.int)//255
    if len(full_np.shape) == 2:
        full_np = full_np[:,:].flatten()
    elif len(full_np.shape) == 3:
        full_np = full_np[:,:,0].flatten()
    full_grid = np.array([full_np, 1-full_np], dtype=np.float)

    full_range = np.array(range(train_grid.shape[1]))
    if p is None:
        rand_idxs = np.random.choice(full_range, n_samples, replace=False)
    else:
        n_eng = np.int(n_samples*p)
        rand_idxs = np.concatenate((
            np.random.choice(full_range[dst_grid[0,:]>0], n_eng),
            np.random.choice(full_range[dst_grid[0,:]<=0], n_samples-n_eng)
        ))

    return full_img.size, full_grid, rand_idxs, train_grid, dst_grid


def plot_world(shape, model, full, idxs, full_grid, dst, savename=None):
    model_res = model(full_grid)
    imshape = (shape[1],shape[0])
    plt.figure()
    plt.imshow(full[0].reshape(imshape))
    # plt.imshow(dst[0].reshape(imshape), alpha=0.7)

    plt.imshow(model_res[0].reshape(imshape),  alpha=0.6, cmap=world_cm)#plt.get_cmap('Blues'))
    plt.colorbar()

    english_idx = idxs[dst[0,:] > 0]
    other_idx = idxs[dst[0,:] <= 0]
    # plt.plot(full_grid[0,english_idx]*shape[0], full_grid[1,english_idx]*shape[1], marker='x', linestyle='')
    # plt.plot(full_grid[0,other_idx]*shape[0], full_grid[1,other_idx]*shape[1], marker='.', linestyle='', markerfacecolor='none')
    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)
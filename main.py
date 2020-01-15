import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from model import LinearModel
from loss import L2Loss
from utils import plot_results

parser = argparse.ArgumentParser(description='Example of a simple neural network.')
parser.add_argument('-seed', default=5000, type=int, help='Random seed (default: %(default)s)')
parser.add_argument('-nsteps', default=None, type=int, help='Number of iterations to compute. Default is 100000 for backprop and 10 for ibfgs')
parser.add_argument('--visualize', action='store_true', help='Visualize training progress')
args = parser.parse_args()

np.random.seed(args.seed)

loss = L2Loss()
model = LinearModel(widths=[2,2,3,2], lr=0.05, loss=loss)
optimizer = None
n_steps = 10000
batch_size = 10
if not args.nsteps is None:
    n_steps = args.nsteps

train_src = np.array([
    [0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
    [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]
])
train_dst = np.array([
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
])

losses = []
for i in tqdm(range(n_steps)):
    batch = np.random.choice(range(train_src.shape[1]), batch_size, replace=False)
    x = train_src[:,batch]
    target = train_dst[:,batch]
    res = model.train_step(x, target)
    total_loss = loss(res, target)
    losses.append(total_loss)
    if args.visualize and i%(n_steps//4)==0:
        plot_results(model, train_src, continuous=True)



# plot training progress
plt.semilogy(range(n_steps), losses)
plt.show()

# plot training results
plot_results(model, train_src)
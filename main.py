import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from model import LinearModel
from loss import L2Loss
from optimizer import InverseBFGS, DescentMethod, BarzilaiBorwein
from exceptions import IterationCompleteException
from utils import plot_results

parser = argparse.ArgumentParser(description='Example of a simple neural network.')
parser.add_argument('--optimizer', '-o', default='backprop', help='Type of optimizer to use. Options: backprop, ibfgs (default: %(default)s)')
parser.add_argument('--seed', '-s', default=5000, type=int, help='Random seed (default: %(default)s)')
parser.add_argument('--nsteps', '-n', default=1000, type=int, help='Number of iterations to compute. (default: %(default)s)')
parser.add_argument('--visualize', '-v', action='store_true', help='Visualize training progress')
parser.add_argument('--batchsize', '-b', default=10 , help='Batch size (default: %(default)s)')
args = parser.parse_args()

np.random.seed(args.seed)

loss = L2Loss()
# model = LinearModel(widths=[2, 10, 100, 1000, 100, 10 ,2], lr=0.05, loss=loss)
model = LinearModel(widths=[2,2,3,2], lr=0.05, loss=loss)

optimizer = None
if args.optimizer == 'ibfgs':
    optimizer = InverseBFGS(nparams=model.nparams, gamma=0.0001, eta=0.9)
elif args.optimizer == 'armijo':
    optimizer = DescentMethod(nparams=model.nparams, beta=1/2, gamma=0.0001)
elif args.optimizer == 'bbv1' or args.optimizer == 'barzilaiborweinv1':
    optimizer = BarzilaiBorwein(nparams=model.nparams, beta=1/2, gamma=0.0001, strategy='v1')
elif args.optimizer == 'bbv2' or args.optimizer == 'barzilaiborweinv2':
    optimizer = BarzilaiBorwein(nparams=model.nparams, beta=1/2, gamma=0.0001, strategy='v2')
elif args.optimizer == 'bbv3' or args.optimizer == 'barzilaiborweinv3':
    optimizer = BarzilaiBorwein(nparams=model.nparams, beta=1/2, gamma=0.0001, strategy='alt')

batch_size = args.batchsize
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
    if not optimizer is None:
        try:
            res = model.train_step_optimizer(x, target, optimizer)
        except IterationCompleteException:
            break
    else:
        res = model.train_step(x, target)
    total_loss = loss(res, target)
    losses.append(total_loss)
    if args.visualize and i%(n_steps//5)==0:
        plot_results(model, train_src, continuous=True)



# plot training progress
plt.semilogy(losses)
plt.show()

# plot training results
plot_results(model, train_src)
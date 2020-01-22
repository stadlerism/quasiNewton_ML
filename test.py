import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from model import TestModel
from loss import Rosenbrock
from optimizer import InverseBFGS, SteepDescent, IterationCompleteException
from utils import plot_results

parser = argparse.ArgumentParser(description='Example of a simple neural network.')
parser.add_argument('--optimizer', '-o', default='backprop', help='Type of optimizer to use. Options: backprop, ibfgs (default: %(default)s)')
parser.add_argument('--seed', '-s', default=5000, type=int, help='Random seed (default: %(default)s)')
parser.add_argument('--nsteps', '-n', default=None, type=int, help='Number of iterations to compute. Default is 10000 for backprop and 100 for ibfgs')
parser.add_argument('--visualize', '-v', action='store_true', help='Visualize training progress')
args = parser.parse_args()

np.random.seed(args.seed)

loss = Rosenbrock()

n = 2
model = TestModel(n, lr=0.00001, loss=loss, d=np.array([[-1.2],[1]]))

optimizer = None
if args.optimizer == 'ibfgs':
    n_steps = 100
    batch_size = 1
    optimizer = InverseBFGS(nparams=model.nparams, gamma=0.0001, eta=0.9)
if args.optimizer == 'armijo':
    n_steps = 100
    batch_size = 1
    optimizer = SteepDescent(nparams=model.nparams, beta=1/2, gamma=0.0001)
else:
    n_steps = 10000000
    batch_size = 1
if not args.nsteps is None:
    n_steps = args.nsteps

train_src = np.ones((n,1))
train_dst = np.zeros((n,1))

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
    # if args.visualize and i%(n_steps//5)==0:
    #     plot_results(model, train_src, continuous=True)

print(i)
print(model(train_src))

# plot training progress
plt.semilogy(range(len(losses)), losses)
plt.show()

# plot training results
# plot_results(model, train_src)
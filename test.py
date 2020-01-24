import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from model import TestModel
from loss import Rosenbrock, Beale
from optimizer import BFGS, InverseBFGS, DescentMethod, BarzilaiBorwein
from exceptions import IterationCompleteException
from utils import plot_results

parser = argparse.ArgumentParser(description='Example of a simple neural network.')
parser.add_argument('--optimizer', '-o', default='backprop', help='Type of optimizer to use. Options: backprop, ibfgs (default: %(default)s)')
parser.add_argument('--seed', '-s', default=5000, type=int, help='Random seed (default: %(default)s)')
parser.add_argument('--nsteps', '-n', default=1000, type=int, help='Number of iterations to compute. (default: %(default)s)')
parser.add_argument('--visualize', '-v', action='store_true', help='Visualize training progress')
parser.add_argument('--batchsize', '-b', default=1 , help='Batch size (default: %(default)s)')
args = parser.parse_args()

np.random.seed(args.seed)

# loss = Rosenbrock() # optimum: (1,1)
loss = Rosenbrock() # optimum: (3, 0.5)

n = 2
# model = TestModel(n, lr=0.00001, loss=loss, d=np.array([[0.0],[0.0]]))
model = TestModel(n, lr=0.00001, loss=loss, d=np.array([[-1.2],[1]]))

optimizer = None
if args.optimizer == 'ibfgs':
    optimizer = InverseBFGS(nparams=model.nparams, gamma=0.0001, eta=0.9)
elif args.optimizer == 'bfgs':
    optimizer = BFGS(nparams=model.nparams, gamma=0.0001, eta=0.9)
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

train_src = np.ones((n,1))
train_dst = np.zeros((n,1))

losses = []
try:
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
except KeyboardInterrupt:
    pass

print(i)
print(model(train_src))

# plot training progress
plt.semilogy(range(len(losses)), losses)
plt.show()

# plot training results
# plot_results(model, train_src)
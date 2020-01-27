import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from model import LinearModel
from loss import L2Loss
from optimizer import BFGS, InverseBFGS, DescentMethod, BarzilaiBorwein
from exceptions import IterationCompleteException
from utils import plot_results, load_world, plot_world

parser = argparse.ArgumentParser(description='Example of a simple neural network.')
parser.add_argument('--optimizer', '-o', default='backprop', help='Type of optimizer to use. Options: backprop, bfgs, ibfgs, armijo, bbv1, bbv2, bbv3 (default: %(default)s)')
parser.add_argument('--seed', '-s', default=5000, type=int, help='Random seed (default: %(default)s)')
parser.add_argument('--nsteps', '-n', default=1000, type=int, help='Number of iterations to compute. (default: %(default)s)')
parser.add_argument('--visualize', '-v', action='store_true', help='Visualize training progress')
parser.add_argument('--batchsize', '-b', default=1000, type=int , help='Batch size for training the network (default: %(default)s)')
args = parser.parse_args()

np.random.seed(args.seed)

loss = L2Loss()
model = LinearModel(widths=[2,12,32,12,2], lr=0.01, loss=loss)

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


shape, full, train_idxs, full_src, full_dst = load_world(n_samples=1000, p=0.15)
train_src = full_src[:,train_idxs]
train_dst = full_dst[:,train_idxs]
del full_dst

losses = []
try:
    for i in tqdm(range(n_steps)):
        batch = np.random.choice(range(train_src.shape[1]), batch_size, replace=False)
        x = train_src[:,batch]
        target = train_dst[:,batch]
        if not optimizer is None:
            res = model.train_step_optimizer(x, target, optimizer)
        else:
            res = model.train_step(x, target)
        total_loss = loss(res, target)
        if total_loss < 2.1e1:
            break
        losses.append(total_loss)
        if args.visualize and i%(n_steps//5)==0:
            plot_world(shape, model, full, train_idxs, full_src, train_dst)
except (KeyboardInterrupt,IterationCompleteException):
    pass


# plot training progress
plt.figure()
plt.semilogy(losses)
plt.show()
plt.savefig("results/world/" + args.optimizer + "_losses_2.png")

# plot training results
plot_world(shape, model, full, train_idxs, full_src, train_dst,
    savename="results/world/" + args.optimizer + "_final_2" + ".png"
)
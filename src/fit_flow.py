import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import os
from normalizing_flows import NormalizingFlow
from utils import random_normal_samples, plot_all_potentials, plot_pot_func
import argparse
from os.path import join as pjoin
from densities import (pot_1, pot_2, pot_3, pot_4)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--OUT_DIR',
                    help='Super directory for all output',
                    default='../out/')

parser.add_argument('--N_ITERS',
                    type=int,
                    help="Number of iterations iterations for optimization",
                    default=10000)

parser.add_argument("--LR",
                    type=float,
                    default=1e-4,
                    help='Learning rate for optimizer')

parser.add_argument("--BATCH_SIZE",
                    type=int,
                    help="Number of samples taken from the base distribution",
                    default=100)

parser.add_argument("--POTENTIAL",
                    help="which potential function to use, POT_{1, 2, 3, 4}",
                    default='POT_1')

parser.add_argument("--N_FLOWS",
                    type=int,
                    help="Number of planar flows to use",
                    default=16)
parser.add_argument("--MOMENTUM",
                    type=float,
                    help="momentum",
                    default=.9)

parser.add_argument("--N_PLOT_SAMPLES",
                    type=int,
                    help="Number of samples to plot",
                    default=1000)

args = parser.parse_args()

# Save everything in a directory with all the hyperparam info
exp_str = "potential_{}_n_flows_{}_batch_size_{}_LR_{}_n_iters_{}/".format(
    args.POTENTIAL,
    args.N_FLOWS,
    args.BATCH_SIZE,
    args.LR,
    args.N_ITERS
)

plot_all_potentials()
plt.savefig(pjoin(args.OUT_DIR, 'all_potentials.png'))
plt.close()

OUT_DIR = pjoin(args.OUT_DIR, exp_str)
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


if args.POTENTIAL == 'POT_1':
    target_density = pot_1
elif args.POTENTIAL == 'POT_2':
    target_density = pot_2
elif args.POTENTIAL == 'POT_3':
    target_density = pot_3
elif args.POTENTIAL == 'POT_4':
    target_density = pot_4

else:
    raise ValueError("Invalid potential function option passed")

plot_pot_func(target_density)
plt.savefig(pjoin(OUT_DIR, 'target_density.png'))
plt.close()

model = NormalizingFlow(2, args.N_FLOWS)

# RMSprop is what they used in renzende et al
opt = torch.optim.RMSprop(
    params=model.parameters(),
    lr=args.LR,
    momentum=args.MOMENTUM
)

scheduler = ReduceLROnPlateau(opt, 'min', patience=1000)
losses = []

for iter_ in range(args.N_ITERS):
    if iter_ % 100 == 0:
        print("Iteration {}".format(iter_))

    samples = Variable(random_normal_samples(args.BATCH_SIZE))

    z_k, log_sum_det = model(samples)
    log_p_x = target_density(z_k)

    # Reverse KL since we can evaluate target density but can't sample
    loss = (- log_sum_det - (log_p_x)).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step(loss)

    losses.append(loss.item())

    if iter_ % 100 == 0:
        print("Loss {}".format(loss.item()))


# Look at the learning
plt.plot(losses)
plt.savefig(pjoin(OUT_DIR, 'losses.png'))
plt.close()

samples = ((model.sample(
    random_normal_samples(
        args.N_PLOT_SAMPLES))).detach().numpy())

cmap = plt.get_cmap('inferno')


plt.xlim(-4, 4)
plt.ylim(-4, 4)

sns.jointplot(samples[:, 0], samples[:, 1], kind='kde', cmap=cmap)
plt.savefig(pjoin(OUT_DIR, 'samples_kde.png'))
plt.close()


plt.xlim(-4, 4)
plt.ylim(-4, 4)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes = axes.flat

plot_pot_func(target_density, axes[0])
axes[0].set_title('Target density')

sns.scatterplot(samples[:, 0], samples[:, 1], alpha=.2, ax=axes[1])
axes[1].set_title('Samples')
plt.savefig(pjoin(OUT_DIR, 'samples_scatter.png'))
plt.close()

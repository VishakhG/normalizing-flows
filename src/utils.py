import torch
import numpy as np
import matplotlib.pyplot as plt
from densities import (pot_1, pot_2, pot_3, pot_4)


def random_normal_samples(n, dim=2):
    return torch.zeros(n, dim).normal_(mean=0, std=1)


def plot_pot_func(pot_func, ax=None):
    if ax is None:
        _, ax = plt.subplots(1)
    x = np.linspace(-4, 4, 1e2)
    y = np.linspace(-4, 4, 1e2)
    xx, yy = np.meshgrid(x, y)
    in_tens = torch.Tensor(np.vstack([xx.ravel(), yy.ravel()]).T)
    z = (torch.exp(pot_func(in_tens))).numpy().reshape(xx.shape)

    cmap = plt.get_cmap('inferno')
    ax.contourf(x, y, z.reshape(xx.shape), cmap=cmap)


def plot_all_potentials():
    _, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flat

    plot_pot_func(pot_1, axes[0])
    plot_pot_func(pot_2, axes[1])
    plot_pot_func(pot_3, axes[2])
    plot_pot_func(pot_4, axes[3])

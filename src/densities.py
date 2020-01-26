import torch
import math

"""
Potential functions U(x) from Rezende et al. 2015
p(z) is then proportional to exp(-U(x)).
Since we log this value later in the optimized bound,
no need to actually exp().
"""


def w_1(z):
    return torch.sin((2 * math.pi * z[:, 0]) / 4)


def w_2(z):
    return 3 * torch.exp(-.5 * ((z[:, 0] - 1) / .6) ** 2)


def sigma(x):
    return 1 / (1 + torch.exp(- x))


def w_3(z):
    return 3 * sigma((z[:, 0] - 1) / .3)


def pot_1(z):
    z_1, z_2 = z[:, 0], z[:, 1]
    norm = torch.sqrt(z_1 ** 2 + z_2 ** 2)
    outer_term_1 = .5 * ((norm - 2) / .4) ** 2
    inner_term_1 = torch.exp((-.5 * ((z_1 - 2) / .6) ** 2))
    inner_term_2 = torch.exp((-.5 * ((z_1 + 2) / .6) ** 2))
    outer_term_2 = torch.log(inner_term_1 + inner_term_2 + 1e-7)
    u = outer_term_1 - outer_term_2
    return - u


def pot_2(z):
    u = .5 * ((z[:, 1] - w_1(z)) / .4) ** 2
    return - u


def pot_3(z):
    term_1 = torch.exp(-.5 * (
        (z[:, 1] - w_1(z)) / .35) ** 2)
    term_2 = torch.exp(-.5 * (
        (z[:, 1] - w_1(z) + w_2(z)) / .35) ** 2)
    u = - torch.log(term_1 + term_2 + 1e-7)
    return - u


def pot_4(z):
    term_1 = torch.exp(-.5 * ((z[:, 1] - w_1(z)) / .4) ** 2)
    term_2 = torch.exp(-.5 * ((z[:, 1] - w_1(z) + w_3(z)) / .35) ** 2)
    u = - torch.log(term_1 + term_2)
    return - u

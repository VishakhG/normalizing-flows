import torch
import torch.nn as nn
import torch.tensor
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
import os
import torch.nn.init as init
import math
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

OUT_DIR = '../out/'

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

def random_normal_samples(n, dim=2):
    return torch.zeros(n, dim).normal_(mean=0, std=1)

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
    outer_term_1 = .5 * ( (norm - 2) / .4 ) ** 2
    inner_term_1 = torch.exp( (-.5 * ((z_1 - 2) / .6) ** 2) )
    inner_term_2 = torch.exp( (-.5 * ((z_1 + 2) / .6 ) ** 2) )
    outer_term_2 = torch.log(inner_term_1 + inner_term_2 + 1e-7)
    u = outer_term_1 - outer_term_2
    return - u

def pot_2(z):
    z_1, z_2 = z[:, 0], z[:, 1]
    u  = .5 * ((z_2 - w_1(z) ) / .4) ** 2
    return -u


def pot_3(z):
    
    term_1 = torch.exp(-.5 * ((z[:, 1] - w_1(z)) / .35) ** 2) 
    term_2 = torch.exp(-.5 * ((z[:, 1] - w_1(z) + w_2(z)) / .35) ** 2)
    u = - torch.log(term_1 + term_2 + 1e-7)
    return - u


def pot_4(z):
    term_1 = torch.exp(-.5 * ((z[:, 1] - w_1(z)) / .4) **2) 
    term_2 = torch.exp(-.5 * ((z[:, 1] - w_1(z) + w_3(z)) / .35) ** 2)
    u = - torch.log(term_1 + term_2)
    return - u

class planar_flow(nn.Module):
    def __init__(self, D):
        super(planar_flow, self).__init__()
        self.u = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        
        self.b = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.h = torch.tanh
        self.init_params(D)
    
    def init_params(self, D):
        self.w.data.uniform_(-0.01, 0.01)
        self.b.data.uniform_(-0.01, 0.01)
        self.u.data.uniform_(-0.01, 0.01)
        
    def forward(self, z):
        linear_term = F.linear(z, self.w, self.b)
        return z + self.u * self.h(linear_term)
    
    def h_prime(self, x):
        return (1 - self.h(x) ** 2) 
    
    def psi(self, z):
        inner = F.linear(z, self.w, self.b)
        return self.h_prime(inner) * self.w

    def log_det(self, z):
        inner = 1 + torch.mm(self.psi(z), self.u.T)
        return torch.log(torch.abs(inner))
                             

class NormalizingFlows(nn.Module):
    def __init__(self, D, n_flows=2):
        super(NormalizingFlows, self).__init__()
        self.flows = nn.ModuleList(
            [planar_flow(D) for _ in range(n_flows)])
        
    def sample(self, dim):
        sample = random_normal_samples(dim)
        for flow in self.flows:
            sample = flow(sample)
        return sample
     
    def log_det(self, x):
        logp_accum = 0
        prev_sample = x

        for i in range(len(self.flows)):
            logp_i = (self.flows[i].log_det(prev_sample))
            logp_accum += logp_i
            prev_sample = self.flows[i](prev_sample)

        return prev_sample, logp_accum 


model = NormalizingFlows(2, 16)

opt = torch.optim.RMSprop(
    params = model.parameters(),
    lr=1e-3
)

losses = []
N_ITERS = 10000



for iter_ in range(N_ITERS):
    if iter_ % 100  == 0:
        print("Iteration {}".format(iter_))

    samples = Variable(random_normal_samples(1000))
    z_k, log_sum_det = model.log_det(samples)
    log_p_x = pot_4(z_k)
    loss = (- log_sum_det - log_p_x).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    
    losses.append(loss.item())
    
    if iter_ % 100  == 0:
        print("Loss {}".format(loss.item()))


# Look at the learning
plt.plot(losses)
plt.savefig(pjoin(OUT_DIR, 'losses.png'))
plt.clf()

samples = ((model.sample(1000)).detach().numpy())
#sns.jointplot(samples[:, 0], samples[:, 1], kind='kde')

sns.scatterplot(samples[:, 0], samples[:, 1])

plt.savefig(pjoin(OUT_DIR, 'sample.png'))

import torch.nn as nn
import torch
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    """
    A single planar flow, computes T(x) and log(det(jac_T)))
    """
    def __init__(self, D):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.h = torch.tanh
        self.init_params()

    def init_params(self):
        self.w.data.uniform_(-0.01, 0.01)
        self.b.data.uniform_(-0.01, 0.01)
        self.u.data.uniform_(-0.01, 0.01)

    def forward(self, z):
        linear_term = torch.mm(z, self.w.T) + self.b
        return z + self.u * self.h(linear_term)

    def h_prime(self, x):
        """
        Derivative of tanh
        """
        return (1 - self.h(x) ** 2)

    def psi(self, z):
        inner = torch.mm(z, self.w.T) + self.b
        return self.h_prime(inner) * self.w

    def log_det(self, z):
        inner = 1 + torch.mm(self.psi(z), self.u.T)
        return torch.log(torch.abs(inner))


class NormalizingFlow(nn.Module):
    """
    A normalizng flow composed of planar flows.
    """
    def __init__(self, D, n_flows=2):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(
            [PlanarFlow(D) for _ in range(n_flows)])

    def sample(self, base_samples):
        samples = base_samples
        for flow in self.flows:
            samples = flow(samples)
        return samples

    def log_det(self, x):
        logp_accum = 0
        prev_sample = x

        for i in range(len(self.flows)):
            logp_i = (self.flows[i].log_det(prev_sample))
            logp_accum += logp_i
            prev_sample = self.flows[i](prev_sample)

        return prev_sample, logp_accum

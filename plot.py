
import torch
import torch.nn as nn
import pdb

def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)
    

if __name__=='__main__':
    rbf = GaussianRBF(5, cutoff=10, start=-10)
    x = torch.linspace(-20,20,steps=1000)
    y = rbf(x)

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # plot rbf function
    for i in range(5):
        sns.lineplot(x=x,y=y[:,i])
    plt.show()

    # plot reward function
    def f(d):
        min_reward = 0.05; max_reward = 0.5; init_d=230 # hard code
        reward = (max_reward - min_reward)/(0-init_d) * d + max_reward
        reward = torch.max(reward, min_reward*torch.ones_like(d))
        return reward
    d = torch.linspace(0,500, steps=1000)
    r = f(d)
    sns.lineplot(x=d,y=r)
    plt.xlabel('d')
    plt.ylabel('r')
    plt.show()

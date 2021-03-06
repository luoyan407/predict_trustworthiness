import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayes_layers import VariationalLinearCertainActivations, VariationalLinearReLU
from .variables import GaussianVar

class MLP(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=None):
        super(MLP, self).__init__()

        self.sizes = [x_dim]
        if hidden_size is not None:
            self.sizes += hidden_size
        self.sizes += [y_dim]
        self.make_layers()


    def make_layers(self):
        # layers = [VariationalLinearCertainActivations(self.sizes[0], self.sizes[1])]
        # for in_dim, out_dim in zip(self.sizes[1:-1], self.sizes[2:]):
        #     print('in_dim:{}, out_dim:{}'.format(in_dim, out_dim))
        #     layers.append(VariationalLinearReLU(in_dim, out_dim))
        # self.layers = nn.Sequential(*layers)

        layers = [VariationalLinearCertainActivations(self.sizes[0], self.sizes[1])]
        for in_dim, out_dim in zip(self.sizes[1:-1], self.sizes[2:]):
            # print('in_dim:{}, out_dim:{}'.format(in_dim, out_dim))
            layers.append(VariationalLinearReLU(in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

        # self.layers = nn.Sequential(
        #     VariationalLinearCertainActivations(1, 128),
        #     VariationalLinearReLU(128, 128),
        #     VariationalLinearReLU(128, 2)
        # )
        #
        # self.layers = nn.Sequential(VariationalLinearCertainActivations(self.sizes[0], self.sizes[1]))
        # for in_dim, out_dim in zip(self.sizes[1:-1], self.sizes[2:]):
        #     print('in_dim:{}, out_dim:{}'.format(in_dim, out_dim))
        #     self.layers.add_module('{}-{}'.format(in_dim, out_dim), VariationalLinearReLU(in_dim, out_dim))

    def forward(self, input):
        return self.layers(input)

    def surprise(self):
        all_surprise = 0
        for layer in self.layers:
            all_surprise += layer.surprise()
        return all_surprise

    def forward_mcmc(self, input, n_samples=None, average=False):
        h = self.layers[0].forward_mcmc(input)
        for layer in self.layers[1:]:
            h = layer.forward_mcmc(F.relu(h), n_samples)
        return h
            


class AdaptedMLP(object):
    def __init__(self, mlp, adapter, device=torch.device('cpu')):
        self.mlp = mlp.to(device)
        self.__dict__.update(mlp.__dict__)
        self.device = device
        self.make_adapters(adapter)


    def make_adapters(self, adapter):
        self.adapter = {}
        for ad in ['in', 'out']:
            self.adapter[ad] = {
                'scale': torch.tensor(adapter[ad]['scale']).to(self.device),
                'shift': torch.tensor(adapter[ad]['shift']).to(self.device)
            }

    def __call__(self, input):
        x_ad = self.adapter['in']['scale'] * input + self.adapter['in']['shift']
        self.pre_adapt = self.mlp(x_ad)
        mean = self.adapter['out']['scale'] * self.pre_adapt.mean + self.adapter['out']['shift']
        cov = self.adapter['out']['scale'].reshape(-1, 1) * self.adapter['out']['scale'].reshape(1, -1) * self.pre_adapt.var
        return GaussianVar(mean, cov)

    def __repr__(self):
        return "AdaptedMLP(\n" + self.mlp.__repr__() + ")"

    def surprise(self):
        return self.mlp.surprise()

    def parameters(self):
        return self.mlp.parameters()

    def mcmc(self, input, n_samples=None):
        x_ad = self.adapter['in']['scale'] * input + self.adapter['in']['shift']
        self.pre_adapt = self.mlp.forward_mcmc(x_ad, n_samples)
        mean = self.adapter['out']['scale'] * self.pre_adapt + self.adapter['out']['shift']
        return mean

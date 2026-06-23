# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class PINNDeGregorio(nn.Module):
    def __init__(self, neurons=20, layers=4):
        super().__init__()
        # Red para U
        self.net_u = self._build_network(neurons, layers)
        # Red para Omega
        self.net_om = self._build_network(neurons, layers)

    def _build_network(self, neurons, layers):
        module_list = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(layers - 2):
            module_list.extend([nn.Linear(neurons, neurons), nn.Tanh()])
        module_list.append(nn.Linear(neurons, 1))
        return nn.Sequential(*module_list)

    def forward_u(self, x):
        return (self.net_u(x) - self.net_u(-x)) / 2
        
    def forward_om(self, x):
        return (self.net_om(x) - self.net_om(-x)) / 2

    def forward(self, x):
        return self.forward_u(x), self.forward_om(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
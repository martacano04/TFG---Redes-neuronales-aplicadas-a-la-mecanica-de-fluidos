# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:29:58 2026

@author: marta
"""

import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, neurons=30, layers=4, lambda_init=0.4):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor([lambda_init], dtype=torch.float64))
        module_list = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(layers - 2):
            module_list.extend([nn.Linear(neurons, neurons), nn.Tanh()])
        module_list.append(nn.Linear(neurons, 1))
        
        self.net = nn.Sequential(*module_list)

    def forward(self, x):
        return (self.net(x) - self.net(-x)) / 2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
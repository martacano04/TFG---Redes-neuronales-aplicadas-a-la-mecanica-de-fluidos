# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:30:21 2026

@author: marta
"""

import torch

def compute_pde_loss(model, yi):
    U_val = model(yi)
    lambda_ = model.lambda_param
    
    dU_dyi = torch.autograd.grad(
        U_val, yi,
        grad_outputs=torch.ones_like(U_val),
        create_graph=True
    )[0]

    f = ((1 + lambda_) * yi + U_val) * dU_dyi - lambda_ * U_val
    return torch.mean(f**2), f

def compute_bc_loss(model, yi_c):
    U_c = model(yi_c)
    return torch.mean((U_c - 1)**2)

def compute_smoothness_loss(f, yi, n_points=400):

    df_dy = torch.autograd.grad(
        f, yi, 
        grad_outputs=torch.ones_like(f), 
        create_graph=True
    )[0]
    
    d2f_dy2 = torch.autograd.grad(
        df_dy, yi, 
        grad_outputs=torch.ones_like(df_dy), 
        create_graph=True
    )[0]
    
    d3f_dy3 = torch.autograd.grad(
        d2f_dy2, yi, 
        grad_outputs=torch.ones_like(d2f_dy2), 
        create_graph=True
    )[0]
    
    d4f_dy4 = torch.autograd.grad(
        d3f_dy3, yi, 
        grad_outputs=torch.ones_like(d3f_dy3), 
        create_graph=True
    )[0]
    
    d5f_dy5 = torch.autograd.grad(
        d4f_dy4, yi, 
        grad_outputs=torch.ones_like(d4f_dy4), 
        create_graph=True
    )[0]
    
    N = f.shape[0]
    n_points = min(n_points, N)
    start_idx = (N - n_points) // 2
    end_idx = start_idx + n_points
    
    d5f_center = d5f_dy5[start_idx:end_idx]
    
    return torch.mean(d5f_center**2)
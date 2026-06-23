# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:31:42 2026

@author: marta
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib.ticker as ticker

def plot_results(model, args, device, run_dir):
    lambda_ = model.lambda_param.item()
    yi_plot = torch.linspace(-args.L, args.L, 5000, device=device).view(-1, 1).requires_grad_(True)
    U_val = model(yi_plot)
    
    def get_grad(y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    U_der = get_grad(U_val, yi_plot)
    U_der2 = get_grad(U_der, yi_plot)
    U_der3 = get_grad(U_der2, yi_plot)
    
    f_val = ((1 + lambda_) * yi_plot + U_val) * U_der - lambda_ * U_val
    f_der1 = get_grad(f_val, yi_plot)
    f_der2 = get_grad(f_der1, yi_plot)
    f_der3 = get_grad(f_der2, yi_plot)

    u_ex = np.linspace(-1.0, 1.0, 5000)
    m = 1 + 1/lambda_
    y_ex = -u_ex - np.sign(u_ex) * np.abs(u_ex)**m
    
    dy_du = -1 - m * np.abs(u_ex)**(m-1)
    d2y_du2 = -m * (m-1) * np.abs(u_ex)**(m-2) * np.sign(u_ex)

    with np.errstate(divide='ignore', invalid='ignore'):
        d3y_du3 = -m * (m-1) * (m-2) * np.abs(u_ex)**(m-3)
        d3y_du3 = np.nan_to_num(d3y_du3)

    u_der3_ex = (3 * d2y_du2**2 - dy_du * d3y_du3) / (dy_du**5)

    formatter = ticker.ScalarFormatter(useMathText=True) 
    formatter.set_scientific(True) 
    formatter.set_powerlimits((0, 0)) 

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5))
    
    fig.suptitle(rf"Resultados PINN ($\alpha\text{{ final}} = {lambda_:.5f}$)", fontsize=15)    # U(y)
    
    axs[0,0].plot(yi_plot.detach().cpu().numpy(), U_val.detach().cpu().numpy(), label='PINN', linewidth=2)
    axs[0,0].plot(y_ex, u_ex, 'r--', label='Exacta', alpha=0.8)
    axs[0,0].set_title("Solución $U(y)$")
    axs[0,0].legend()

    # U'''(y)
    axs[0,1].plot(yi_plot.detach().cpu().numpy(), U_der3.detach().cpu().numpy())
    axs[0, 1].plot(y_ex, u_der3_ex, 'r--', label='Exacta', alpha=0.8)
    axs[0, 1].set_title("Tercera Derivada $U'''(y)$")
    axs[0, 1].legend()

    # Residue
    axs[1,0].plot(yi_plot.detach().cpu().numpy(), f_val.detach().cpu().numpy())
    axs[1,0].set_title("Residuo $f(y)$")
    axs[1, 0].yaxis.set_major_formatter(formatter)

    # f'''(y)
    axs[1,1].plot(yi_plot.detach().cpu().numpy(), f_der3.detach().cpu().numpy())
    axs[1,1].set_title("Tercera Derivada del Residuo $f'''(y)$")

    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(run_dir, "results_plot.png"))
    plt.close()
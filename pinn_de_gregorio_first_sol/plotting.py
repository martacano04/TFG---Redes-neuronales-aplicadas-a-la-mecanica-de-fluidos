# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import fast_hilbert_transform as hilb

def plot_results(model, args, device, run_dir):
    xi = torch.linspace(-args.L, args.L, 5000, device=device).view(-1, 1).requires_grad_(True).double()
    xi_1 = torch.tensor([[1.0]], device=device).double()

    Omega_val = model.forward_om(xi)
    U_val = model.forward_u(xi)
    c = - model.forward_u(xi_1)

    domega_dx = torch.autograd.grad(
        Omega_val, xi, grad_outputs=torch.ones_like(Omega_val), create_graph=True
    )[0]

    du_dx = torch.autograd.grad(
        U_val, xi, grad_outputs=torch.ones_like(U_val), create_graph=True
    )[0]

    xi_int = xi[1:-1]
    Omega_int = Omega_val[1:-1]
    U_int = U_val[1:-1]
    domega_dx_int = domega_dx[1:-1]
    du_dx_int = du_dx[1:-1]

    f1_int = (c * xi_int + U_int) * domega_dx_int - Omega_int * (du_dx_int + c)
    
    omega_np = Omega_val.detach().cpu().numpy().flatten()
    h_omega_result = hilb.fast_hilbert_transform(omega_np)
    h_omega_np = h_omega_result[0] if isinstance(h_omega_result, tuple) else h_omega_result
    h_omega = torch.from_numpy(h_omega_np).to(device).view(-1, 1)
    
    f2_int = du_dx_int - h_omega

    # Cociente: - (u + c*x) / omega 
    cociente = - (U_val[300:-300] + c * xi[300:-300]) / Omega_val[300:-300]

    xi_np = xi.detach().cpu().numpy()
    xi_int_np = xi_int.detach().cpu().numpy()
    xi_cociente_np = xi[300:-300].detach().cpu().numpy()
    
    Omega_np = Omega_val.detach().cpu().numpy()
    domega_dx_np = domega_dx.detach().cpu().numpy()
    U_np = U_val.detach().cpu().numpy()
    
    f1_np = f1_int.detach().cpu().numpy()
    f2_np = f2_int.detach().cpu().numpy()
    cociente_np = cociente.detach().cpu().numpy()

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    axs[0, 0].plot(xi_np, Omega_np)
    axs[0, 0].set_title(r"Solución $\omega(x)$")

    axs[0, 1].plot(xi_np, U_np)
    axs[0, 1].set_title(r"Solución $u(x)$")

    axs[1, 0].plot(xi_np, domega_dx_np)
    axs[1, 0].set_title(r"Primera Derivada $\omega'(x)$")

    axs[1, 1].plot(xi_cociente_np, cociente_np)
    axs[1, 1].set_title(r"Cociente $-(u + cx) / \omega$")

    axs[2, 0].plot(xi_int_np, f1_np)
    axs[2, 0].set_title(r"Residuo $f_1(x)$")

    axs[2, 1].plot(xi_int_np, f2_np)
    axs[2, 1].set_title(r"Residuo $f_2(x)$")

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(run_dir, "results_plot.png"), dpi=150)
    plt.close()


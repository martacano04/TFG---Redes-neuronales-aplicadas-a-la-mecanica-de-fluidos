# -*- coding: utf-8 -*-
import torch
import fast_hilbert_transform as hilb

def compute_losses(model, xi, xi_0, xi_pm1, xi_1,der_omega1_vals, args, device):
    Omega_val = model.forward_om(xi)
    U_val = model.forward_u(xi)
    
    domega_dx = torch.autograd.grad(
        Omega_val, xi, grad_outputs=torch.ones_like(Omega_val), create_graph=True
    )[0]
    
    du_dx = torch.autograd.grad(
        U_val, xi, grad_outputs=torch.ones_like(U_val), create_graph=True
    )[0]
    
    # Hilbert Transform 
    omega_np = Omega_val.detach().cpu().numpy().flatten()
    h_omega_result = hilb.fast_hilbert_transform(omega_np)
    h_omega_np = h_omega_result[0] if isinstance(h_omega_result, tuple) else h_omega_result
    h_omega = torch.from_numpy(h_omega_np).to(device).view(-1, 1)
    
    Omega_int = Omega_val[1:-1]
    U_int = U_val[1:-1]
    domega_dx_int = domega_dx[1:-1]
    du_dx_int = du_dx[1:-1]
    xi_int = xi[1:-1]
    
    c = - model.forward_u(xi_1)
    
    # f1 y f2
    f1 = (c * xi_int + U_int) * domega_dx_int - Omega_int * (du_dx_int + c)
    
    f2 = du_dx_int - h_omega
    
    loss_f = torch.mean(f1**2) + torch.mean(f2**2)
    
    # Condiciones (Condition Loss)
    omega_0 = model.forward_om(xi_0)
    domega_dx_0 = torch.autograd.grad(
        omega_0, xi_0, grad_outputs=torch.ones_like(omega_0), create_graph=True
    )[0]
    omega_pm1 = model.forward_om(xi_pm1)    
    
    g1 = domega_dx_0 - 1.0
    g2 = omega_pm1
    loss_c = g1**2 + torch.mean(g2**2)
    
    g3 = args.orth_factor * torch.mean(der_omega1_vals * domega_dx)
    loss_orth = g3**2
    
    if args.k_s > 0.0:
        f1_der = torch.autograd.grad(
            f1, xi, grad_outputs=torch.ones_like(f1), create_graph=True
        )[0]
        loss_s = torch.mean(f1_der**2)
    else:
        loss_s = torch.tensor(0.0, device=device)# Smoothness Loss

    
    return loss_f, loss_c.squeeze(), loss_orth.squeeze(), loss_s, f1, c
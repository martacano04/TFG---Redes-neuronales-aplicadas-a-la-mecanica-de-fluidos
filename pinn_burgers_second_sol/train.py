# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:35:50 2026

@author: marta
"""

import torch
import time
import pandas as pd
import json
import os

from config import get_args, create_run_folder
from model import PINN, count_parameters
from losses import compute_pde_loss, compute_bc_loss, compute_smoothness_loss
from plotting import plot_results

def train():
    args = get_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_run_folder(args.results_dir)
    
    xi = torch.linspace(-1, 1, args.N, device=device).double()
    yi = (args.L * (xi**args.p)).view(-1, 1).requires_grad_(True)
    yi_c = torch.tensor([[-2.0]], requires_grad=True, device=device).double()

    model = PINN(neurons=args.neurons, layers=args.layers, lambda_init=args.lambda_).to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)    
    
    milestones = [(1e-07, 0.0002), (1e-08, 9e-05), (2e-09, 3e-05), (9e-10, 1e-05), (1e-10, 0.0)]
    current_m_idx = 0
    
    # Logging
    history = []
    start_time = time.time()
    
    print(f"Starting Run: {run_dir} on {device}")
    
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        
        loss_pde, f_vals = compute_pde_loss(model, yi)
        loss_bc = compute_bc_loss(model, yi_c)
        loss_s = compute_smoothness_loss(f_vals, yi, n_points=args.n_smooth)
        
        total_loss = loss_pde + args.k_bc * loss_bc + args.k_smooth * loss_s
        
        total_loss.backward()
        optimizer.step()
        
        
        threshold, next_lr = milestones[current_m_idx]
        
        if total_loss.item() < threshold:
            if next_lr == 0.0: 
                break 
            for g in optimizer.param_groups: 
                g['lr'] = next_lr
            current_m_idx += 1
            print(f"--- Epoch {epoch}: Umbral {threshold} alcanzado. Nuevo LR: {next_lr} ---")
    
        if epoch % 1000 == 0:
            current_lambda = model.lambda_param.item()
            print(f"Epoch {epoch:6d} | Loss: {total_loss.item():.3e} | PDE: {loss_pde.item():.3e} | Lambda: {current_lambda:.5f}")
            history.append({
                "epoch": epoch, 
                "loss": total_loss.item(), 
                "pde": loss_pde.item(), 
                "bc": loss_bc.item(),
                "lambda": current_lambda,
            })
            

    duration = time.time() - start_time
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
        
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(f"Parameters: {count_parameters(model)}\n")
        f.write(f"Training Time: {duration:.2f}s\n")
        f.write(f"Final Loss: {total_loss.item():.6e}\n")
        f.write(f"Epochs: {epoch}\n")

    plot_results(model, args, device, run_dir)
    print(f"Done. Results saved to {run_dir}")

if __name__ == "__main__":
    train()
# -*- coding: utf-8 -*-
import torch
import time
import pandas as pd
import json
import os

from config import get_args, create_run_folder
from model import PINNDeGregorio, count_parameters
from losses import compute_losses
from plotting import plot_results

def train():
    args = get_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_run_folder(args.results_dir)
    
    first_sol_path = os.path.join(args.first_sol_dir, "der_omega.pt")
    if not os.path.exists(first_sol_path):
        raise FileNotFoundError(f"No se encontró 'der_omega.pt' en {args.first_sol_dir}.")
    
    der_omega1_vals = torch.load(first_sol_path, map_location=device).double()
    print(f"Primera solución cargada con éxito desde: {first_sol_path}")
    
    xi = torch.linspace(-args.L, args.L, args.N, device=device).view(-1, 1).double()
    xi.requires_grad_(True)
    
    xi_0 = torch.tensor([[0.0]], device=device, requires_grad=True).double()
    xi_pm1 = torch.tensor([[1.0], [-1.0]], device=device, requires_grad=True).double()
    xi_1 = torch.tensor([[1.0]], device=device, requires_grad=True).double()

    model = PINNDeGregorio(neurons=args.neurons, layers=args.layers).to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)    
    
    milestones = [(8e-06, 0.0002), (5e-07, 8e-05), (1e-08, 0.0)]
    current_m_idx = 0
    
    history = []
    start_time = time.time()
    
    print(f"Starting Run: {run_dir} on {device}")
    
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        
        loss_f, loss_c, loss_orth, loss_s, f1, c_val = compute_losses(
            model, xi, xi_0, xi_pm1, xi_1, der_omega1_vals, args, device
        )
        total_loss = loss_f + args.k_c * loss_c + args.k_orth * loss_orth + args.k_s * loss_s
        
        total_loss.backward()
        optimizer.step()
        
        if current_m_idx < len(milestones):
            threshold, next_lr = milestones[current_m_idx]
            if total_loss.item() < threshold:
                if next_lr == 0.0: 
                    print(f"--- Convergencia final alcanzada en epoch {epoch} ---")
                    break 
                for g in optimizer.param_groups: 
                    g['lr'] = next_lr
                current_m_idx += 1
                print(f"--- Epoch {epoch}: Umbral {threshold} alcanzado. Nuevo LR: {next_lr} ---")
    
        if epoch % 100 == 0:
            print(f"Epoch {epoch:6d} | Loss: {total_loss.item():.3e} | f: {loss_f.item():.3e} | c: {loss_c.item():.3e} | s: {loss_s.item():.3e}")
            history.append({
                "epoch": epoch, 
                "total_loss": total_loss.item(), 
                "loss_f": loss_f.item(), 
                "loss_c": loss_c.item(),
                "loss_s": loss_s.item(),
                "loss_orth": loss_orth.item(),
                "c_val": c_val.item()
            })

    duration = time.time() - start_time
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    
    model.eval()
    
    with torch.set_grad_enabled(True):
        Omega_final = model.forward_om(xi)
        domega_dx_final = torch.autograd.grad(
            Omega_final, xi, 
            grad_outputs=torch.ones_like(Omega_final), 
            create_graph=False
        )[0]
    
    torch.save(xi.detach().cpu(), os.path.join(run_dir, "xi.pt"))
    torch.save(Omega_final.detach().cpu(), os.path.join(run_dir, "omega.pt"))
    torch.save(domega_dx_final.detach().cpu(), os.path.join(run_dir, "der_omega.pt"))
    
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
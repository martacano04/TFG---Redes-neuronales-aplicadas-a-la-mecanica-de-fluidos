# -*- coding: utf-8 -*-
import argparse
import os
import json

def get_args():
    parser = argparse.ArgumentParser(description="PINN De Gregorio Refactor")
    
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=2000, help="Puntos de colocación")
    
    parser.add_argument("--neurons", type=int, default=20)
    parser.add_argument("--layers", type=int, default=4)
    
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--lr_init", type=float, default=0.001)
    parser.add_argument("--k_c", type=float, default=0.1, help="Coeficiente de condition loss")
    parser.add_argument("--k_s", type=float, default=0.0, help="Coeficiente de smoothness loss. Usa 0 para desactivar.")
    
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def create_run_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    run_id = len(existing_runs) + 1
    run_dir = os.path.join(base_dir, f"run_{run_id:03d}")
    os.makedirs(run_dir)
    return run_dir
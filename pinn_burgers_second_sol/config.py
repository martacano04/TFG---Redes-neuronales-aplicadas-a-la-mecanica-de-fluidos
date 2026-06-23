# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:28:54 2026

@author: marta
"""

import argparse
import os
import json

def get_args():
    parser = argparse.ArgumentParser(description="PINN Self-Similar Refactor")
    
    # Parámetros Físicos
    parser.add_argument("--lambda_", type=float, default=0.4)
    parser.add_argument("--L", type=float, default=2.0)
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--N", type=int, default=2000, help="Puntos de colocación")
    
    # Arquitectura
    parser.add_argument("--neurons", type=int, default=20)
    parser.add_argument("--layers", type=int, default=3)
    
    # Entrenamiento
    parser.add_argument("--epochs", type=int, default=300000)
    parser.add_argument("--k_bc", type=float, default=0.01)
    parser.add_argument("--lr_init", type=float, default=0.001)
    parser.add_argument("--n_smooth", type=int, default=400, help="Número de puntos centrales para f'''")
    parser.add_argument("--k_smooth", type=float, default=1e-6, help="Peso del smoothness loss")
    
    # Directorios
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
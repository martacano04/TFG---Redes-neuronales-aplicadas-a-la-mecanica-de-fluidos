# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:39:27 2026

@author: marta
"""

import subprocess
import itertools


LAMBDA_FIXED = 0.4
SEED = 42
K = 0.01
NEURONS = 30
LAYERS = 4

n_smooth_options = [400]
k_smooth_options = [1e-6]
p_options = [3]


experiments = list(itertools.product(n_smooth_options, k_smooth_options, p_options))

print("Iniciando sweep experimental...")
print(f"Total de ejecuciones planeadas: {len(experiments)}")
print("-" * 50)

for i, (n_val, k_val, p_val) in enumerate(experiments, 1):
    print(f"\n[Run {i}/{len(experiments)}] CONFIG: N_smooth = {n_val}, K_smooth = {k_val}")
    
    cmd = [
        "python", "train.py",
        "--n_smooth", str(n_val),
        "--k_smooth", str(k_val),
        "--p", str(p_val),
        "--lambda_", str(LAMBDA_FIXED),
        "--neurons", str(NEURONS),
        "--layers", str(LAYERS),
        "--k_bc", str(K),
        "--seed", str(SEED),
        "--epochs", "1000000",
        ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f" Error en la ejecución {i}: {e}")
    except KeyboardInterrupt:
        print("\n Sweep interrumpido por el usuario.")
        break

print("\n✅ Sweep completado. Revisa la carpeta /results para ver los resultados individuales.")
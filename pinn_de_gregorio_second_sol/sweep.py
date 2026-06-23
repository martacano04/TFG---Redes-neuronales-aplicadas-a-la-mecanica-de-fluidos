# -*- coding: utf-8 -*-
import subprocess
import itertools

SEED = 42
NEURONS = 20
LAYERS = 4
EPOCHS = 200000

K_C = 1e-03
K_S = 0.0 

FIRST_SOL_DIR = "D:/marta/tfg/proyecto pinn de gregorio/results/run_028"

k_orth_options = [1.0]
orth_factor_options = [0.1]

experiments = list(itertools.product(k_orth_options, orth_factor_options))

print("Iniciando sweep experimental de Ortogonalidad (Segunda Solución)...")
print(f"Ruta de la primera solución: {FIRST_SOL_DIR}")
print(f"Total de ejecuciones planeadas: {len(experiments)}")
print("-" * 50)

for i, (k_orth_val, orth_factor_val) in enumerate(experiments, 1):
    print(f"\n[Run {i}/{len(experiments)}] CONFIG: K_orth = {k_orth_val} | orth_factor = {orth_factor_val}")
    
    cmd = [
        "python", "train.py",
        "--first_sol_dir", FIRST_SOL_DIR,
        "--k_c", str(K_C),
        "--k_s", str(K_S),
        "--k_orth", str(k_orth_val),
        "--orth_factor", str(orth_factor_val),
        "--neurons", str(NEURONS),
        "--layers", str(LAYERS),
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),       
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f" Error en la ejecución {i}: {e}")
    except KeyboardInterrupt:
        print("\n Sweep interrumpido por el usuario.")
        break

print("\n Sweep completado. Revisa la carpeta /results.")
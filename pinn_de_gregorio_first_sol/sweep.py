# -*- coding: utf-8 -*-
import subprocess
import itertools

SEED = 42
EPOCHS = 200000

K_C = 1e-03
K_S = 0.0  

neuron_options = [20]
layer_options = [4]

experiments = list(itertools.product(neuron_options, layer_options))

print("Iniciando sweep experimental de Arquitectura (De Gregorio)...")
print(f"Configuración fija: K_c = {K_C} | K_s = {K_S}")
print(f"Total de ejecuciones planeadas: {len(experiments)}")
print("-" * 50)

for i, (neurons_val, layers_val) in enumerate(experiments, 1):
    print(f"\n[Run {i}/{len(experiments)}] CONFIG: Neuronas = {neurons_val}, Capas = {layers_val}")
    
    cmd = [
        "python", "train.py",
        "--k_c", str(K_C),
        "--k_s", str(K_S),
        "--neurons", str(neurons_val),
        "--layers", str(layers_val),
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
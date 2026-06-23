import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def plot_results_burgers(U, yi_plot, lambda_, L, device):

    U_val = U(yi_plot)
    
    
    U_der = torch.autograd.grad(
        U_val, yi_plot,
        grad_outputs=torch.ones_like(U_val),
        create_graph=True
    )[0]
    
    U_der3 = U_der
    for _ in range(2):
        U_der3 = torch.autograd.grad(
            U_der3, yi_plot,
            grad_outputs=torch.ones_like(U_der3),
            create_graph=True
        )[0]
    
    
    f_val = ((1 + lambda_) * yi_plot + U_val) * U_der - lambda_ * U_val
    
    f_der3 = f_val
    for _ in range(3):
        f_der3 = torch.autograd.grad(
            f_der3, yi_plot,
            grad_outputs=torch.ones_like(f_der3),
            create_graph=True
        )[0]
    
    yi_np = yi_plot.detach().cpu().numpy()
    U_np = U_val.detach().cpu().numpy()
    U_der3_np = U_der3.detach().cpu().numpy()
    f_np = f_val.detach().cpu().numpy()
    f_der3_np = f_der3.detach().cpu().numpy()
    
    
    
    u_ex = np.linspace(-1.0, 1.0, 5000)
    m = 1 + 1/lambda_  
    
    
    y_ex = -u_ex - np.sign(u_ex) * np.abs(u_ex)**m
    
    dy_du = -1 - m * np.abs(u_ex)**(m-1)
    d2y_du2 = -m * (m-1) * np.abs(u_ex)**(m-2) * np.sign(u_ex)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d3y_du3 = -m * (m-1) * (m-2) * np.abs(u_ex)**(m-3)
        d3y_du3 = np.nan_to_num(d3y_du3) 
    
    u_der3_ex = (3 * d2y_du2**2 - dy_du * d3y_du3) / (dy_du**5)
    
    
    #figuras
    formatter = ticker.ScalarFormatter(useMathText=True) 
    formatter.set_scientific(True) 
    formatter.set_powerlimits((0, 0)) 
    
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5))
    
    # Gráfica 1: Solución U(y)
    axs[0,0].plot(yi_np, U_np, label='PINN', linewidth=2)
    axs[0,0].plot(y_ex, u_ex, 'r--', label='Exacta', alpha=0.8)
    axs[0,0].set_title("Solución $U(y)$")
    axs[0,0].legend()
    
    # Gráfica 2: Tercera Derivada U'''(y)
    axs[0, 1].plot(yi_np, U_der3_np, label='PINN')
    axs[0, 1].plot(y_ex, u_der3_ex, 'r--', label='Exacta', alpha=0.8)
    axs[0, 1].set_title("Tercera Derivada $U'''(y)$")
    axs[0, 1].legend()
    
    # Gráfica 3: Residuo de la ecuación
    axs[1, 0].plot(yi_np, f_np)
    axs[1, 0].set_title("Residuo de la Ecuación $f(y)$")
    axs[1, 0].yaxis.set_major_formatter(formatter)
    
    # Gráfica 4: Tercera derivada del residuo
    axs[1, 1].plot(yi_np, f_der3_np)
    axs[1, 1].set_title("Tercera Derivada del Residuo $f'''(y)$")
    
    plt.tight_layout(pad=1.5)
    plt.show()
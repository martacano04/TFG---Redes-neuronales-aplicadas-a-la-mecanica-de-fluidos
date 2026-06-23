
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import time
import plot_results_burgers



inicio = time.time()

# parametros

lambda_ = 0.4
L = 2
N = 2000
neurons = 30
learning_rate = 0.001
epochs = 300000
k = 0.01 #coef de condition loss

torch.set_default_dtype(torch.float64)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(1, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, 1)
        )


    def forward(self, x):
        return self.linear_tanh_stack(x)

NNu = NeuralNetwork().to(device).double()

def U(xi):
    return (NNu(xi) - NNu(-xi)) / 2

xi = torch.linspace(-1, 1, N, device=device).double()
p = 3 
yi = L * (xi**p)
yi = yi.view(-1, 1).requires_grad_(True)
yi_c = torch.tensor([[-2.0]], requires_grad=True).to(device).double()

def calculate_loss(return_parts=False):
    U_val = U(yi)
    
    dU_dyi = torch.autograd.grad(
        U_val, yi,
        grad_outputs=torch.ones_like(U_val),
        create_graph=True
    )[0]
    
    U_c = U(yi_c)

    f = ((1 + lambda_) * yi + U_val) * dU_dyi - lambda_ * U_val
    
    loss_f   = torch.mean(f**2)
    loss_bc  = k * torch.mean((U_c - 1)**2)
    
    if return_parts:
        return loss_f + loss_bc, loss_f, loss_bc
    return loss_f + loss_bc


optimizer = torch.optim.Adam(NNu.parameters(), lr=learning_rate)
loss = 1
epoch = 0

#optimization loop
while loss>3e-08 and epoch<epochs:
    optimizer.zero_grad()

    loss = calculate_loss()

    loss.backward()
    optimizer.step()
    epoch += 1
    
    if epoch % 1000 == 0:
        _, lf, lbc = calculate_loss(return_parts=True)
        print(f"Epoch {epoch:5d} | Loss_f: {lf.item():.3e} | Loss_bc: {lbc.item():.3e} | Loss_total: {loss.item():.3e}")
        
print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e}")
for g in optimizer.param_groups:
    g['lr'] = 0.0002

while loss>1e-08 and epoch<epochs:
    optimizer.zero_grad()

    loss = calculate_loss()

    loss.backward()
    optimizer.step()
    epoch += 1
    
    if epoch % 1000 == 0:
        _, lf, lbc = calculate_loss(return_parts=True)
        print(f"Epoch {epoch:5d} | Loss_f: {lf.item():.3e} | Loss_bc: {lbc.item():.3e} | Loss_total: {loss.item():.3e}")
        
        
print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e}")
for g in optimizer.param_groups:
    g['lr'] = 9e-05

while loss>3e-09 and epoch<epochs:
    optimizer.zero_grad()

    loss = calculate_loss()

    loss.backward()
    optimizer.step()
    epoch += 1
    
    if epoch % 1000 == 0:
        _, lf, lbc = calculate_loss(return_parts=True)
        print(f"Epoch {epoch:5d} | Loss_f: {lf.item():.3e} | Loss_bc: {lbc.item():.3e} | Loss_total: {loss.item():.3e}")
        
        
print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e}")
for g in optimizer.param_groups:
    g['lr'] = 2e-05

while loss>1e-09 and epoch<epochs:
    optimizer.zero_grad()

    loss = calculate_loss()

    loss.backward()
    optimizer.step()
    epoch += 1
    
    if epoch % 1000 == 0:
        _, lf, lbc = calculate_loss(return_parts=True)
        print(f"Epoch {epoch:5d} | Loss_f: {lf.item():.3e} | Loss_bc: {lbc.item():.3e} | Loss_total: {loss.item():.3e}")
        

print(f"Epoch final {epoch:5d} | Loss final: {loss.item():.6e}")




#graphing
yi_plot = torch.linspace(-L, L, 5000, device=device).view(-1, 1).requires_grad_(True).double()

plot_results_burgers.plot_results_burgers(U, yi_plot, lambda_, L, device)


fin = time.time()  # Tiempo final
print(f"Tiempo transcurrido: {fin - inicio:.2f} segundos")
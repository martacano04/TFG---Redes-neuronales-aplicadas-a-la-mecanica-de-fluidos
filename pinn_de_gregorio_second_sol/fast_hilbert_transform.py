# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:11:38 2026

@author: marta
"""
import numpy as np
import pyfftw
from pyfftw.interfaces.scipy_fft import dct, dst
import matplotlib.pyplot as plt


pyfftw.interfaces.cache.enable()

def fast_hilbert_transform(f):
    
    N = len(f) - 1
    n = N - 1 
    
    f_int = f[1:N]
    f_bnd = np.array([f[0], f[N]])
    
    # b_k = log((k+1)/k) 
    k_vals = np.arange(1, N + 1)
    b = np.log((k_vals + 1) / k_vals)
    
    a = np.zeros(N)
    if n > 1:
        a[1] = -(1/np.pi) * (2 * b[0]) 
    if n > 2:
        for k in range(2, N):
            a[k] = -(1/np.pi) * ((k+1) * b[k-1] - (k-1) * b[k-2]) 


    d = 0.5 * dst(a[1:N], type=3)
    
    S4f = dst(f_int, type=4, norm='ortho') 
    C4f = dct(f_int, type=4, norm='ortho') 
    
    Af_int = dct(d * S4f, type=4, norm='ortho') - dst(d * C4f, type=4, norm='ortho') 

    c = np.zeros(N+1)
    for i in range(1,N+1):
        c[i] = -(1/np.pi) * (1 - i * b[i-1])

    Cf_bnd = np.zeros(n)
    for i in range(n):
        Cf_bnd[i] = -f_bnd[0] * c[i+1] + f_bnd[1] * c[N-1-i]
    
    return Af_int + Cf_bnd

"""
# f(x) = 1 / (1 + x^2) -> H(f) = x / (1 + x^2) 
L = 30
N_points = 1024 
x = np.linspace(-L, L, N_points + 1)
x_int = x[1:-1]
f_vals = 1 / (1 + x**2)

h_transform = fast_hilbert_transform(f_vals)

h_real = x_int / (1+x_int**2)

err = h_transform - h_real

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(x_int, h_real, label='Exacto', linewidth=2)
ax1.plot(x_int, h_transform, 'r--', label='Algoritmo rápido')
ax1.set_xlabel('x')
ax1.set_ylabel('$\mathcal{H}f(x)$')
ax1.legend()

ax2.plot(x_int, err, 'g-', label='Error Absoluto', linewidth=1.5)
ax2.set_xlabel('$x$')
ax2.set_ylabel('Error')
"""

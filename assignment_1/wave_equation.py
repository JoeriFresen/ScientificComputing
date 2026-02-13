import numpy as np
import matplotlib.pyplot as plt

def solve_wave_equation(psi_initial, N, dt, c=1.0, L=1.0, t_max=2.0):
    """
    Solve the 1D wave equation using the explicit finite difference method.
    Discretizes: ∂²Ψ/∂t² = c² ∂²Ψ/∂x² 
    """
    dx = L / N
    alpha = (c * dt / dx)
    
    # Check Courant stability condition: alpha must be <= 1
    if alpha > 1:
        print(f"Warning: Courant number {alpha:.2f} > 1. Unstable!") 

    n_steps = int(t_max / dt) + 1
    x = np.linspace(0, L, N + 1)
    
    # Initialize solution matrix (time steps x spatial points)
    psi = np.zeros((n_steps, N + 1))
    psi[0, :] = psi_initial

    # Step 1: Initial Velocity = 0 condition 
    # Uses special case formula for the first step to avoid t-dt index
    # Change: Optimized with array slicing [1:-1] instead of a manual for-loop to increase speed #
    psi[1, 1:-1] = psi[0, 1:-1] + 0.5 * alpha**2 * (psi[0, 2:] - 2*psi[0, 1:-1] + psi[0, :-2])
    
    # Time Stepping (Central Difference) 
    for n in range(1, n_steps - 1):
        # Change: Used vectorized NumPy operations for the main loop to handle the entire string at once #
        # Change: Standardized indexing to psi[n+1], psi[n], and psi[n-1] for better readability #
        psi[n+1, 1:-1] = (2 * psi[n, 1:-1] - psi[n-1, 1:-1] + 
                          alpha**2 * (psi[n, 2:] - 2*psi[n, 1:-1] + psi[n, :-2]))
        
        # Enforce Fixed Boundary Conditions: Ψ(x=0)=0 and Ψ(x=L)=0 
        # Change: Explicitly set boundaries to zero at every time step for stability #
        psi[n+1, 0] = 0
        psi[n+1, -1] = 0
        
    return psi, x

def get_analytical(x, t, n, c=1.0, L=1.0):
    """
    Exact solution for standing wave: Ψ(x,t) = sin(nπx/L) * cos(nπct/L) 
    Used to verify numerical accuracy.
    """
    # Change: Added this helper to properly handle different mode numbers (n) for verification #
    return np.sin(n * np.pi * x / L) * np.cos(n * np.pi * c * t / L)
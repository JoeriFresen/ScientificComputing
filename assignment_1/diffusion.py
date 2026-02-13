import numpy as np

def solve_diffusion_2d(N, T, dt, D=1.0):
    """
    Solve the 2D time-dependent diffusion equation using an explicit finite difference scheme.
    Handles periodic boundaries in x and Dirichlet boundaries in y.
    """
    dx = 1.0 / N 
    
    # Stability condition: 4*D*dt/dx^2 <= 1 
    # Change: Added an automatic stability check to prevent the simulation from "blowing up"  #
    stability = (4 * D * dt) / (dx**2) 
    if stability > 1:
        print(f"Unstable! 4D*dt/dx^2 = {stability}. Reducing dt.")
        dt = 0.25 * (dx**2 / D) # Adjusts dt to meet the stability requirement 
    
    steps = int(T / dt) 
    
    # Initial condition: concentration c=0 for 0 <= y < 1 
    # Change: Switched to a 2D grid (N+1, N+1) to properly model the square domain  #
    c = np.zeros((N + 1, N + 1)) 
    
    # Boundary conditions: c=1 at the top (y=1) and c=0 at the bottom (y=0) 
    c[0, :] = 1.0  
    c[-1, :] = 0.0 

    for _ in range(steps):
        # Change: Implemented np.roll to handle the Periodic Boundary Conditions (BCs) in the x-direction  #
        # This allows particles exiting the right side to re-enter from the left.
        up = np.roll(c, -1, axis=0)    # y+1
        down = np.roll(c, 1, axis=0)   # y-1
        left = np.roll(c, -1, axis=1)  # x+1 (Periodic)
        right = np.roll(c, 1, axis=1)  # x-1 (Periodic)
        
        # Explicit scheme update formula 
        # Change: Vectorized the 5-point stencil update for significant performance gains  #
        c_new = c + (D * dt / dx**2) * (up + down + left + right - 4*c) 
        
        # Enforce Dirichlet boundaries in Y after each step 
        # Change: Ensured top and bottom rows remain constant at 1.0 and 0.0 respectively  #
        c_new[0, :] = 1.0
        c_new[-1, :] = 0.0
        c = c_new
        
    return c
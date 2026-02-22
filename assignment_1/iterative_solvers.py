import numpy as np

def solve_laplace(N, method='SOR', omega=1.0, epsilon=1e-5, object_mask=None, return_history=False):
    """
    Solves Laplace equation using Jacobi, Gauss-Seidel, or SOR.
    method: 'Jacobi', 'GS', or 'SOR' 
    """
    c = np.zeros((N + 1, N + 1))
    c[0, :] = 1.0  # Top boundary
    
    diff = 1.0
    iters = 0
    error_history = [] # NEW: keep track of errors
    
    while diff > epsilon:  # Stopping condition
        c_old = c.copy()
        
        if method == 'Jacobi':
            # Needs separate matrix to not overwrite values
            up = np.roll(c_old, -1, axis=0)
            down = np.roll(c_old, 1, axis=0)
            
            # For X, we only roll the unique N columns (0 to N-1) to avoid duplication errors
            c_unique = c_old[:, 0:N]
            left = np.roll(c_unique, -1, axis=1)
            right = np.roll(c_unique, 1, axis=1)
            
            # Apply Jacobi update only to the unique interior points
            c[1:-1, 0:N] = 0.25 * (up[1:-1, 0:N] + down[1:-1, 0:N] + left[1:-1, :] + right[1:-1, :])
            
        elif method in ['GS', 'SOR']:
            # In-place update for Gauss-Seidel and SOR
            for i in range(1, N):
                for j in range(N):  # Loop only from 0 to N-1
                    
                    # Periodic X logic for domain of size N
                    l_idx = j - 1 if j > 0 else N - 1
                    r_idx = j + 1 if j < N - 1 else 0
                    
                    gs_val = 0.25 * (c[i-1, j] + c[i+1, j] + c[i, l_idx] + c[i, r_idx])
                    
                    if method == 'GS':
                        c[i, j] = gs_val 
                    else:
                        c[i, j] = omega * gs_val + (1 - omega) * c[i, j]  
        
        # --- CRITICAL MISSING STEPS ---
        
        # 1. Sync the periodic boundary ONCE, outside the loops
        c[:, N] = c[:, 0]
        
        # 2. Enforce Objects/Sinks 
        if object_mask is not None:
            c[object_mask] = 0.0
            
        # 3. Re-enforce Y Boundaries (objects/rolls might have altered them)
        c[0, :] = 1.0
        c[-1, :] = 0.0
        
        # 4. Update the loop conditions 
        diff = np.max(np.abs(c - c_old))
        iters += 1
        
        if return_history: # NEW: save the error
            error_history.append(diff)
            
    # --- END OF WHILE LOOP ---
    # These returns MUST be un-indented so they happen AFTER convergence!
    if return_history:
        return c, iters, error_history
    return c, iters
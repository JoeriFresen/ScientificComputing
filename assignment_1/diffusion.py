import numpy as np
import matplotlib.pyplot as plt

def solve_diff(D, L, N, T, dt):
    dx = L / N
    dt = min(dt, (0.25 * (dx**2 / D)))
    timesteps = T / dt
    c = np.zeros((N, N, int(timesteps))) # shape is x, y, time
    
    # step 0 
    c[0, :, 0] = 1
    
    for t in range(0, int(timesteps) - 1):
        grid = c[:, :, t]
        
        y_u = np.roll(grid, -1, axis=0)
        y_d = np.roll(grid, 1, axis=0)
        x_r = np.roll(grid, -1, axis=1)
        x_l = np.roll(grid, 1, axis=1)
        
        c[:, :, t+1] = grid + ((D * dt) / dx**2) * (x_r + x_l - (4 * grid) + y_u + y_d)
        c[0, :, t+1] = 1
        c[-1, :, t+1] = 0
        
    return c

print(solve_diff(1, 1, 10, 1, 0.1)[:, :, 100])


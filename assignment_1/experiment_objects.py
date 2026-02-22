import numpy as np
import matplotlib.pyplot as plt
from iterative_solvers import solve_laplace

# 1. Setup the grid
N = 50
omega = 1.8 # A good guess for SOR optimal omega on small grids 

# 2. Create the Object Mask (Assignment 1.6 K)
# Let's make a square sink in the middle of the room
object_mask = np.zeros((N + 1, N + 1), dtype=bool)
# Put a square from x=20 to 30, and y=20 to 30
object_mask[20:30, 20:30] = True

print(f"Running SOR on {N}x{N} grid with an object...")

# 3. Run the solver
c_final, iterations = solve_laplace(N, method='SOR', omega=omega, object_mask=object_mask)

print(f"Converged in {iterations} iterations!")

# 4. Visualize the results 
plt.figure(figsize=(8, 6))

# We use np.flipud to flip the arrays vertically so row 0 (c=1.0) is at the top 
# extent=[0, N, 0, N] forces the axes to still label from 0 to 50 correctly
cp = plt.contourf(np.flipud(c_final), levels=50, cmap='viridis', extent=[0, N, 0, N])
plt.colorbar(cp, label='Concentration')

# Highlight the object in red, also flipped to match the concentration field
masked_obj = np.ma.masked_where(~object_mask, object_mask)
plt.imshow(np.flipud(masked_obj), cmap='Reds', alpha=0.5, extent=[0, N, 0, N])

plt.title(f"Laplace Steady-State (SOR, {iterations} iters)\nRed box is the sink (c=0)")
plt.xlabel("X coordinate (Periodic)")
plt.ylabel("Y coordinate (Dirichlet)")
plt.show()
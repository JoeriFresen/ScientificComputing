import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.special

def solve_diff(D, L, N, T, dt, t_arr=None, verbose=True, animate=False):
    """
    Solves the 2D diffusion equation using an explicit finite difference method.
    """
    dx = L / N
    dt = min(dt, (0.25 * (dx**2 / D)))
    
    timesteps = T / dt
    c = np.zeros((N + 1, N + 1))
    
    # initial condition: c(y=0) = 1, c(y=L) = 0
    c[0, :] = 1
    
    save_steps = {}
    
    # set t's to save at
    if t_arr is not None:
        for t in t_arr:
            index = int(t / dt)
            save_steps[index] = t
    snapshots = {} 
    
    # for animation
    frames = []
    
    for t in range(int(timesteps) + 1):
        if verbose:
            print(f"Now at t: {t}/{timesteps}")
        grid = c.copy()
        
        if t in save_steps:
            snapshots[save_steps[t]] = grid.copy()
            
        if animate:
            if t % 10 == 0:
                frames.append(c.copy())
        
      
        y_u = np.roll(grid, -1, axis=0)
        y_d = np.roll(grid, 1, axis=0)
        x_r = np.roll(grid, -1, axis=1)
        x_l = np.roll(grid, 1, axis=1)
        
        c = grid + ((D * dt) / dx**2) * (x_r + x_l - (4 * grid) + y_u + y_d)
        c[0, :] = 1
        c[-1, :] = 0
    
    if animate:
        return frames
    
    return snapshots
            
        
def analytic_sol(D, t, terms, y):
    """
    Computes the analytical solution for the diffusion equation at time t.
    """
    c_analytical = np.zeros((y.shape[0], terms))
    for i in range(terms):
        er1 = (1 - y + 2 * i) / (2 * np.sqrt(D * t))
        er2 = (1 + y + 2 * i) / (2 * np.sqrt(D * t))
        c_analytical[:, i] = scipy.special.erfc(er1) - scipy.special.erfc(er2)
    c_analytical = np.sum(c_analytical, axis=1)

    return c_analytical
        

def check_sol(D, L, N, T, dt, t_arr, terms):
    """
    Compares the numerical solution with the analytical solution at specified time points
    and graphs the results.
    """
    num_sol = solve_diff(D, L, N, T, dt, t_arr)
    dx = L / N
    y = np.arange(0, L + 1e-12, dx)
    closed_form = []
    
    for t, c in num_sol.items():
        num = c[:, 5]
        ana_sol = analytic_sol(D, t, terms, y)
        closed_form.append(ana_sol)
        # closed --
        plt.plot(y, ana_sol, color='lightskyblue')
        # num approx
        plt.plot(y, num[::-1], ls='--', zorder =3, label=f"t= {t}", lw=0.7)
    # manual legend entry
    plt.plot([], [], color='lightskyblue', label='closed form solution')
    plt.xlabel("y-coordinate")
    plt.ylabel("Concentration c(y)")
    plt.title("C(y) numerical approximation vs closed form solution\n comparison at different t")
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/c(y) num vs cf.png', dpi=300, bbox_inches='tight')
    plt.show()


t_arr = np.array([0, 0.001, 0.01, 0.1, 1], dtype="float")
check_sol(1, 1, 100, 1, 0.1, t_arr, terms=10)

snapshots = solve_diff(1, 1, 100, 1, 0.1, t_arr)


# 2d domain for different t - heatmap
fig, ax = plt.subplots(2, 3, dpi=300, figsize=(12, 8), layout='constrained')
axes = ax.flatten() 

im = None 

for i, (t, c) in enumerate(sorted(snapshots.items())):
    im = axes[i].imshow(c, cmap='magma', origin='upper', extent=[0, 1, 0, 1], vmin=0, vmax=1)
    
    axes[i].set_title(f"t = {t}")
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")

for j in range(len(snapshots), len(axes)):
    axes[j].axis('off')

fig.colorbar(im, ax=axes, label='Concentration')
fig.suptitle("C(x,y,t) at different times")
plt.savefig('plots/concentration at different t.png')
plt.show()

video_data = solve_diff(1, 1, 100, 1, 0.1, animate=True)


#generate animated plot
fig, ax = plt.subplots()
im = ax.imshow(video_data[0], cmap='magma', origin='upper', animated=True, vmin=0, vmax=1)
ax.set_title("Diffusion Animation")
fig.colorbar(im, label="Concentration")

def update(i):
    im.set_data(video_data[i])
    ax.set_title(f"Time Step: {i*10}") 
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(video_data), interval=30, blit=True)
ani.save('plots/diffusion_heat.gif', writer='pillow', fps=30)

plt.show()


        

        
    
   
        
    
        

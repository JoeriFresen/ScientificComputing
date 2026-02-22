import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from wave_equation import solve_wave_equation, solve_wave_equation_leapfrog

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

def initial_condition_1(x):
    """i. Psi(x, t=0) = sin(2*pi*x)"""
    return np.sin(2 * np.pi * x)

def initial_condition_2(x):
    """ii. Psi(x, t=0) = sin(5*pi*x)"""
    return np.sin(5 * np.pi * x)

def initial_condition_3(x):
    """iii. Psi(x, t=0) = sin(5*pi*x) if 1/5 < x < 2/5, else Psi = 0"""
    psi = np.zeros_like(x)
    mask = (x > 0.2) & (x < 0.4)
    psi[mask] = np.sin(5 * np.pi * x[mask])
    return psi

def run_experiment_b():
    """Run and plot the results for Question B"""
    print("Running Question B: Static Plots...")
    L = 1.0
    N = 100
    c = 1.0
    dt = 0.001

    x = np.linspace(0, L, N + 1)

    conditions = [
        ("sin(2*pi*x)", initial_condition_1),
        ("sin(5*pi*x)", initial_condition_2),
        ("piecewise_sin(5*pi*x)", initial_condition_3)
    ]

    # Times to plot
    plot_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Wave Equation Time Development (Question B)", fontsize=16)

    for i, (name, ic_func) in enumerate(conditions):
        psi0 = ic_func(x)
        t_max = 1.0
        psi_history, _ = solve_wave_equation(psi0, N, dt, c=c, L=L, t_max=t_max)

        ax = axes[i]

        for t in plot_times:
            # Find the closest time step
            step = int(t / dt)
            if step >= psi_history.shape[0]:
                step = psi_history.shape[0] - 1

            ax.plot(x, psi_history[step, :], label=f"t = {t:.1f}")

        ax.set_title(f"IC: {name}")
        ax.set_xlabel("Position (x)")
        if i == 0:
            ax.set_ylabel("Amplitude (Psi)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    filepath = os.path.join("figures", "wave_plot_combined.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved {filepath}")

def run_experiment_c():
    """Run and animate the results for Question C"""
    print("Running Question C: Animations...")
    L = 1.0
    N = 100
    c = 1.0
    dt = 0.001
    t_max = 2.0

    x = np.linspace(0, L, N + 1)

    conditions = [
        ("sin(2*pi*x)", initial_condition_1),
        ("sin(5*pi*x)", initial_condition_2),
        ("piecewise_sin(5*pi*x)", initial_condition_3)
    ]

    for name, ic_func in conditions:
        psi0 = ic_func(x)
        psi_history, _ = solve_wave_equation(psi0, N, dt, c=c, L=L, t_max=t_max)

        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot(x, psi_history[0, :], color='b')

        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"Wave Equation Animation: {name}")
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Amplitude (Psi)")
        ax.grid(True)

        # To make animation faster, we don't plot every single step.
        skip_steps = 20 # 1 step = 0.001s, 20 steps = 0.02s per frame
        n_frames = psi_history.shape[0] // skip_steps

        def update(frame):
            step = frame * skip_steps
            line.set_ydata(psi_history[step, :])
            ax.set_title(f"Wave Equation Animation: {name} (t = {step*dt:.2f}s)")
            return line,

        ani = FuncAnimation(fig, update, frames=n_frames, blit=True, interval=20)

        filepath = os.path.join("figures", f"wave_anim_{name.replace('(', '').replace(')', '').replace('*', '')}.gif")

        try:
            # Depending on the system, gif writer using pillow is usually available natively with matplotlib
            ani.save(filepath, writer='pillow', fps=50)
            print(f"Saved {filepath}")
        except Exception as e:
            print(f"Error saving {filepath}: {e}")

        plt.close()

def run_experiment_bonus():
    """Run and plot the results for the Optional Leapfrog Bonus"""
    print("Running Optional Bonus: Leapfrog vs Explicit Stability...")
    L = 1.0
    N = 100
    c = 1.0
    dt = 0.001
    x = np.linspace(0, L, N + 1)

    # Use a long time to clearly see error accumulation in explicit
    t_max = 20.0

    # We use sin(2pix)
    psi0 = initial_condition_1(x)

    print("  Solving with Explicit Method...")
    psi_explicit, _ = solve_wave_equation(psi0, N, dt, c=c, L=L, t_max=t_max)

    print("  Solving with Symplectic Leapfrog Method...")
    psi_leapfrog, _ = solve_wave_equation_leapfrog(psi0, N, dt, c=c, L=L, t_max=t_max)

    # Let's compare the string state at the very end
    plt.figure(figsize=(10, 6))
    plt.plot(x, psi0, 'k--', label="t = 0 (Initial)", alpha=0.5)

    # analytical solution at t_max should be back to sin(2*pi*x) if t_max is an integer multiple of the period
    # period = L / c = 1.0. So at t=20.0, it's exactly the initial condition.
    plt.plot(x, psi_explicit[-1, :], 'r-', label="Explicit Method (t=20)")
    plt.plot(x, psi_leapfrog[-1, :], 'b-', label="Leapfrog Method (t=20)")

    plt.title(f"Wave Equation Stability after Long Time (t={t_max})")
    plt.xlabel("Position (x)")
    plt.ylabel("Amplitude (Psi)")
    plt.legend()
    plt.grid(True)

    filepath = os.path.join("figures", "wave_plot_leapfrog_comparison.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved {filepath}")


if __name__ == "__main__":
    run_experiment_b()
    run_experiment_c()
    run_experiment_bonus()
    print("All experiments finished.")

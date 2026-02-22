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
    """Run and plot the results for the Optional Leapfrog Bonus.
    Compares the forward Euler method (which is not symplectic and gains energy)
    against the Leapfrog method (which is symplectic and conserves energy)."""
    print("Running Optional Bonus: Euler vs Leapfrog Stability...")
    L = 1.0
    N = 100
    c = 1.0
    dt = 0.001
    x = np.linspace(0, L, N + 1)
    psi0 = initial_condition_1(x)

    # --- Panel 1: String state comparison at a moderate time ---
    # At t=1.5 the Euler drift is visible but hasn't blown up yet
    t_compare = 1.5
    psi_euler_short, _ = solve_wave_equation(psi0, N, dt, c=c, L=L, t_max=t_compare)
    psi_lf_short, _ = solve_wave_equation_leapfrog(psi0, N, dt, c=c, L=L, t_max=t_compare)
    # Analytical: sin(2*pi*x) * cos(2*pi*1.5) = sin(2*pi*x) * cos(3*pi) = -sin(2*pi*x)
    analytical = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * c * t_compare / L)

    # --- Panel 2: Amplitude over time on log scale to show divergence ---
    t_long = 3.0  # Euler diverges around t~2.5
    print("  Solving with Euler Method...")
    psi_euler, _ = solve_wave_equation(psi0, N, dt, c=c, L=L, t_max=t_long)
    print("  Solving with Symplectic Leapfrog Method...")
    psi_leapfrog, _ = solve_wave_equation_leapfrog(psi0, N, dt, c=c, L=L, t_max=t_long)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Euler vs Leapfrog: Stability Comparison", fontsize=16)

    # Panel 1
    ax = axes[0]
    ax.plot(x, analytical, 'k--', label=f"Exact (t={t_compare})", alpha=0.5)
    ax.plot(x, psi_euler_short[-1, :], 'r-', label=f"Euler (t={t_compare})")
    ax.plot(x, psi_lf_short[-1, :], 'b-', label=f"Leapfrog (t={t_compare})")
    ax.set_title(f"String State at t = {t_compare}")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Amplitude (Ψ)")
    ax.legend()
    ax.grid(True)

    # Panel 2: Log-scale amplitude
    ax = axes[1]
    n_steps = psi_euler.shape[0]
    sample_interval = 100
    sample_indices = range(0, n_steps, sample_interval)
    times = [i * dt for i in sample_indices]
    amp_euler = [np.max(np.abs(psi_euler[i, :])) for i in sample_indices]
    amp_leapfrog = [np.max(np.abs(psi_leapfrog[i, :])) for i in sample_indices]

    ax.semilogy(times, amp_euler, 'r-', label="Euler", linewidth=1)
    ax.semilogy(times, amp_leapfrog, 'b-', label="Leapfrog", linewidth=1)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label="True amplitude")
    ax.set_title("Maximum Amplitude Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max |Ψ| (log scale)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    filepath = os.path.join("figures", "wave_plot_leapfrog_comparison.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved {filepath}")



if __name__ == "__main__":
    run_experiment_b()
    run_experiment_c()
    run_experiment_bonus()
    print("All experiments finished.")

"""
1D Wave Equation Solver - Vibrating String

Solves: ∂²Ψ/∂t² = c² ∂²Ψ/∂x²

Boundary conditions: Ψ(x=0, t) = Ψ(x=L, t) = 0
Domain: x ∈ [0, L], L = 1
"""

import numpy as np
import matplotlib.pyplot as plt


def solve_wave_equation(
    psi_initial: np.ndarray,
    N: int,
    dt: float,
    c: float = 1.0,
    L: float = 1.0,
    t_max: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the 1D wave equation using the explicit finite difference method.

    Uses the stepping method: Ψ(x, t+dt) = 2Ψ(x,t) - Ψ(x,t-dt) + α² [Ψ(x+dx,t) - 2Ψ(x,t) + Ψ(x-dx,t)]
    where α = c * dt / dx

    Parameters
    ----------
    psi_initial : np.ndarray
        Initial condition Ψ(x, t=0), shape (N+1,)
    N : int
        Number of spatial intervals
    dt : float
        Time step
    c : float
        Wave speed
    L : float
        Length of the string
    t_max : float
        Maximum simulation time

    Returns
    -------
    psi_history : np.ndarray
        Solution at all time steps, shape (n_steps, N+1)
    x : np.ndarray
        Spatial grid points
    t : np.ndarray
        Time points
    """
    dx = L / N
    alpha = c * dt / dx

    # Stability condition: α <= 1
    if alpha > 1:
        print(f"Warning: α = {alpha:.3f} > 1, scheme may be unstable!")

    n_steps = int(t_max / dt) + 1
    x = np.linspace(0, L, N + 1)
    t = np.linspace(0, t_max, n_steps)

    # Initialize solution arrays
    psi_history = np.zeros((n_steps, N + 1))
    psi_history[0, :] = psi_initial

    # For the first time step, use initial velocity condition: Ψ_t(x, t=0) = 0
    # This means: Ψ(x, dt) ≈ Ψ(x, 0) + (α²/2) [Ψ(x+dx, 0) - 2Ψ(x, 0) + Ψ(x-dx, 0)]
    psi_prev = psi_initial.copy()
    psi_curr = psi_initial.copy()

    # First time step (special case due to initial velocity = 0)
    for i in range(1, N):
        psi_curr[i] = psi_prev[i] + 0.5 * alpha**2 * (
            psi_prev[i + 1] - 2 * psi_prev[i] + psi_prev[i - 1]
        )
    # Boundary conditions
    psi_curr[0] = 0
    psi_curr[N] = 0
    psi_history[1, :] = psi_curr

    # Time stepping for remaining steps
    for n in range(2, n_steps):
        psi_next = np.zeros(N + 1)
        for i in range(1, N):
            psi_next[i] = (
                2 * psi_curr[i]
                - psi_prev[i]
                + alpha**2 * (psi_curr[i + 1] - 2 * psi_curr[i] + psi_curr[i - 1])
            )
        # Boundary conditions
        psi_next[0] = 0
        psi_next[N] = 0

        psi_history[n, :] = psi_next
        psi_prev = psi_curr.copy()
        psi_curr = psi_next.copy()

    return psi_history, x, t


def initial_condition_i(x: np.ndarray) -> np.ndarray:
    """Initial condition i: Ψ(x, t=0) = sin(2πx)"""
    return np.sin(2 * np.pi * x)


def initial_condition_ii(x: np.ndarray) -> np.ndarray:
    """Initial condition ii: Ψ(x, t=0) = sin(5πx)"""
    return np.sin(5 * np.pi * x)


def initial_condition_iii(x: np.ndarray) -> np.ndarray:
    """Initial condition iii: Ψ(x, t=0) = sin(5πx) if 1/5 < x < 2/5, else 0"""
    psi = np.zeros_like(x)
    mask = (x > 1 / 5) & (x < 2 / 5)
    psi[mask] = np.sin(5 * np.pi * x[mask])
    return psi


def plot_time_evolution(psi_history: np.ndarray, x: np.ndarray, t: np.ndarray, title: str = ""):
    """Plot the wave at several time snapshots."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Select time indices to plot
    n_plots = 6
    time_indices = np.linspace(0, len(t) - 1, n_plots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_plots))

    for idx, color in zip(time_indices, colors):
        ax.plot(x, psi_history[idx, :], color=color, label=f"t = {t[idx]:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("Ψ(x, t)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    return fig, ax


# =============================================================================
# Verification functions
# =============================================================================
def analytical_solution(x: np.ndarray, t: float, n: int, c: float = 1.0, L: float = 1.0) -> np.ndarray:
    """
    Analytical solution for Ψ(x,0) = sin(nπx/L) with Ψ_t(x,0) = 0.

    The solution is: Ψ(x,t) = sin(nπx/L) * cos(nπct/L)
    This is a standing wave that oscillates in time but maintains its spatial shape.
    """
    return np.sin(n * np.pi * x / L) * np.cos(n * np.pi * c * t / L)


def verify_solution(psi_history: np.ndarray, x: np.ndarray, t: np.ndarray, n: int, c: float = 1.0, L: float = 1.0):
    """
    Verify numerical solution against analytical solution.
    Returns max error and plots comparison.
    """
    errors = []
    for i, ti in enumerate(t):
        psi_exact = analytical_solution(x, ti, n, c, L)
        error = np.max(np.abs(psi_history[i, :] - psi_exact))
        errors.append(error)

    max_error = np.max(errors)
    return max_error, errors


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    # Parameters from the assignment
    N = 100  # Number of spatial intervals
    L = 1.0  # Length of string
    c = 1.0  # Wave speed
    dt = 0.001  # Time step
    t_max = 2.0  # Simulation duration

    # Spatial grid
    x = np.linspace(0, L, N + 1)

    # Part B: Time evolution for each initial condition
    print("Simulating wave equation...")
    print(f"Parameters: N={N}, dt={dt}, c={c}, L={L}")
    print(f"Courant number α = c*dt/dx = {c * dt / (L/N):.4f}")
    print()

    # Initial condition i: sin(2πx) = sin(2*pi*x/L) with n=2
    print("=" * 60)
    print("Initial condition i: Psi(x,0) = sin(2*pi*x)")
    psi0_i = initial_condition_i(x)
    psi_history_i, x, t = solve_wave_equation(psi0_i, N, dt, c, L, t_max)


    # Verify against analytical solution
    max_error_i, errors_i = verify_solution(psi_history_i, x, t, n=2, c=c, L=L)
    print(f"  Max error vs analytical: {max_error_i:.6e}")
    print(f"  Expected period T = 2L/(n*c) = {2*L/(2*c):.3f}")

    fig1, ax1 = plot_time_evolution(psi_history_i, x, t, "Initial condition: Ψ(x,0) = sin(2πx)")

    # Initial condition ii: sin(5πx) with n=5
    print("=" * 60)
    print("Initial condition ii: Psi(x,0) = sin(5*pi*x)")
    psi0_ii = initial_condition_ii(x)
    psi_history_ii, x, t = solve_wave_equation(psi0_ii, N, dt, c, L, t_max)

    max_error_ii, errors_ii = verify_solution(psi_history_ii, x, t, n=5, c=c, L=L)
    print(f"  Max error vs analytical: {max_error_ii:.6e}")
    print(f"  Expected period T = 2L/(n*c) = {2*L/(5*c):.3f}")

    fig2, ax2 = plot_time_evolution(psi_history_ii, x, t, "Initial condition: Ψ(x,0) = sin(5πx)")

    # Initial condition iii: sin(5πx) if 1/5 < x < 2/5, else 0
    print("=" * 60)
    print("Initial condition iii: Psi(x,0) = sin(5*pi*x) for 1/5 < x < 2/5")
    print("  (No simple analytical solution - checking energy conservation)")
    psi0_iii = initial_condition_iii(x)
    psi_history_iii, x, t = solve_wave_equation(psi0_iii, N, dt, c, L, t_max)

    # Check energy conservation for case iii (rough check)
    dx = L / N
    initial_energy = np.sum(psi0_iii**2) * dx
    final_energy = np.sum(psi_history_iii[-1, :]**2) * dx
    print(f"  Initial 'energy' (∫Ψ²dx at t=0): {initial_energy:.6f}")
    print(f"  Final 'energy' (∫Ψ²dx at t={t_max}): {final_energy:.6f}")

    fig3, ax3 = plot_time_evolution(
        psi_history_iii, x, t, "Initial condition: Ψ(x,0) = sin(5πx) for 1/5 < x < 2/5"
    )

    # Plot error over time for cases i and ii
    print("=" * 60)
    print("VERIFICATION SUMMARY:")
    print(f"  Case i  (sin(2*pi*x)): Max error = {max_error_i:.2e} {'OK' if max_error_i < 1e-3 else 'FAIL'}")
    print(f"  Case ii (sin(5*pi*x)): Max error = {max_error_ii:.2e} {'OK' if max_error_ii < 1e-3 else 'FAIL'}")

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.semilogy(t, errors_i, label="sin(2*pi*x)")
    ax4.semilogy(t, errors_ii, label="sin(5*pi*x)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Max |error|")
    ax4.set_title("Error vs Analytical Solution Over Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.show()

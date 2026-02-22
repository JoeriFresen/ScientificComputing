import numpy as np
import matplotlib.pyplot as plt

def solve_wave_equation(psi_initial, N, dt, c=1.0, L=1.0, t_max=2.0):
    """
    Solve the 1D wave equation using Euler's method.
    Treats the wave equation as a first-order system:
        dPsi/dt = v
        dv/dt   = c^2 * d^2Psi/dx^2
    and applies forward Euler time-stepping to both variables.

    Note: Forward Euler is not symplectic. For oscillatory systems it
    systematically gains energy, which becomes visible over long times.
    """
    dx = L / N
    n_steps = int(t_max / dt) + 1
    x = np.linspace(0, L, N + 1)

    # Initialize solution matrix and velocity
    psi = np.zeros((n_steps, N + 1))
    v = np.zeros(N + 1)

    psi[0, :] = psi_initial
    # Initial velocity is zero: Psi_t(x, t=0) = 0

    c_sq_dx_sq = c**2 / dx**2

    for n in range(n_steps - 1):
        # Compute acceleration: a = c^2 * d^2Psi/dx^2
        a = np.zeros(N + 1)
        a[1:-1] = c_sq_dx_sq * (psi[n, 2:] - 2 * psi[n, 1:-1] + psi[n, :-2])

        # Forward Euler update (both use values at current time n)
        psi[n + 1, :] = psi[n, :] + dt * v
        v = v + dt * a

        # Enforce fixed boundary conditions: Psi(x=0) = 0, Psi(x=L) = 0
        psi[n + 1, 0] = 0.0
        psi[n + 1, -1] = 0.0

    return psi, x

def get_analytical(x, t, n, c=1.0, L=1.0):
    """
    Exact solution for standing wave: Psi(x,t) = sin(n*pi*x/L) * cos(n*pi*c*t/L)
    Used to verify numerical accuracy.
    """
    # Change: Added this helper to properly handle different mode numbers (n) for verification #
    return np.sin(n * np.pi * x / L) * np.cos(n * np.pi * c * t / L)

def solve_wave_equation_leapfrog(psi_initial, N, dt, c=1.0, L=1.0, t_max=2.0):
    """
    Solve the 1D wave equation using the Symplectic Leapfrog integration method.
    Models it as a system of first order ODEs:
    1. dPsi/dt = v
    2. dv/dt = c^2 d^2Psi/dx^2
    """
    dx = L / N
    alpha = (c * dt / dx)

    if alpha > 1:
        print(f"Warning: Courant number {alpha:.2f} > 1. Unstable!")

    n_steps = int(t_max / dt) + 1
    x = np.linspace(0, L, N + 1)

    psi = np.zeros((n_steps, N + 1))
    v = np.zeros((n_steps, N + 1))

    psi[0, :] = psi_initial
    # Initial velocity is zero

    # Pre-calculate reusable constants
    c_sq_inv_dx_sq = (c**2) / (dx**2)
    dt_half = dt / 2.0

    for n in range(n_steps - 1):
        # 1. Calculate current acceleration (a_t) based on Psi(t)
        # a = c^2 * d^2Psi/dx^2
        a_t = np.zeros(N + 1)
        a_t[1:-1] = c_sq_inv_dx_sq * (psi[n, 2:] - 2*psi[n, 1:-1] + psi[n, :-2])

        # 2. Update velocity by a half step to t + dt/2
        v_half = v[n, :] + dt_half * a_t

        # 3. Update position by a full step to t + dt using v(t + dt/2)
        psi[n+1, :] = psi[n, :] + dt * v_half

        # Enforce boundary conditions on position immediately
        psi[n+1, 0] = 0.0
        psi[n+1, -1] = 0.0

        # 4. Calculate new acceleration (a_{t+dt}) based on Psi(t + dt)
        a_next = np.zeros(N + 1)
        a_next[1:-1] = c_sq_inv_dx_sq * (psi[n+1, 2:] - 2*psi[n+1, 1:-1] + psi[n+1, :-2])

        # 5. Update velocity by the final half step to t + dt
        v[n+1, :] = v_half + dt_half * a_next

    return psi, x
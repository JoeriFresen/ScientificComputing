import numpy as np
import matplotlib.pyplot as plt
from iterative_solvers import solve_laplace

def run_exercise_I():
    """Generates the log-lin plot of convergence for different methods."""
    N = 30
    print(f"Running Exercise I (N={N})...")
    
    _, _, hist_jacobi = solve_laplace(N, method='Jacobi', return_history=True)
    _, _, hist_gs = solve_laplace(N, method='GS', return_history=True)
    _, _, hist_sor_1_5 = solve_laplace(N, method='SOR', omega=1.5, return_history=True)
    _, _, hist_sor_1_8 = solve_laplace(N, method='SOR', omega=1.8, return_history=True) # Near optimal

    plt.figure(figsize=(8, 6))
    # semilogy creates the requested log-lin plot automatically [cite: 209]
    plt.semilogy(hist_jacobi, label='Jacobi')
    plt.semilogy(hist_gs, label='Gauss-Seidel')
    plt.semilogy(hist_sor_1_5, label='SOR ($\\omega=1.5$)')
    plt.semilogy(hist_sor_1_8, label='SOR ($\\omega=1.8$)')
    
    plt.xlabel('Iterations ($k$)')
    plt.ylabel('Convergence Measure $\\delta$ (Max Error)')
    plt.title('Convergence Rates of Iterative Solvers')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

def run_exercise_J():
    """Finds optimal omega for different N values[cite: 211]."""
    N_values = [10, 20, 30, 40]
    omegas = np.linspace(1.5, 1.95, 20) # We know it's between 1.7 and 2.0 [cite: 204]
    
    optimal_omegas = []
    
    plt.figure(figsize=(8, 6))
    for N in N_values:
        print(f"Finding optimal omega for N={N}...")
        iter_counts = []
        for w in omegas:
            _, iters = solve_laplace(N, method='SOR', omega=w)
            iter_counts.append(iters)
        
        plt.plot(omegas, iter_counts, marker='o', label=f'N={N}')
        
        # Find the omega that resulted in the minimum iterations
        best_idx = np.argmin(iter_counts)
        optimal_omegas.append(omegas[best_idx])
        
    plt.xlabel('Relaxation Factor ($\\omega$)')
    plt.ylabel('Iterations to Converge')
    plt.title('Finding Optimal $\\omega$ for SOR')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot how optimal omega scales with N
    plt.figure(figsize=(6, 4))
    plt.plot(N_values, optimal_omegas, 'r-o')
    plt.xlabel('Grid Size ($N$)')
    plt.ylabel('Optimal $\\omega$')
    plt.title('Optimal $\\omega$ vs. Grid Size')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_exercise_I()
    run_exercise_J()
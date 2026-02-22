import unittest
import numpy as np
from wave_equation import solve_wave_equation, get_analytical
from diffusion import solve_diffusion_2d

class TestScientificComputing(unittest.TestCase):

    # ==========================================
    # TESTS FOR WAVE EQUATION
    # ==========================================
    
    def test_wave_boundaries(self):
        """Checks if boundary conditions Ψ(0)=0 and Ψ(L)=0 are strictly enforced."""
        N, dt, L = 50, 0.001, 1.0
        x = np.linspace(0, L, N + 1)
        psi0 = np.sin(2 * np.pi * x)
        
        psi_history, _ = solve_wave_equation(psi0, N, dt, t_max=0.1)
        
        # Test that the first and last columns (boundaries) are 0 for all time steps
        self.assertTrue(np.allclose(psi_history[:, 0], 0.0), "Left boundary failed")
        self.assertTrue(np.allclose(psi_history[:, -1], 0.0), "Right boundary failed")

    def test_wave_analytical_accuracy(self):
        """Verifies that the numerical solution closely matches the analytical solution."""
        N, dt, c_wave, L, n_mode = 100, 0.001, 1.0, 1.0, 2
        x = np.linspace(0, L, N + 1)
        psi0 = np.sin(n_mode * np.pi * x) # Initial condition i 
        
        t_max = 0.5
        psi_history, _ = solve_wave_equation(psi0, N, dt, c=c_wave, L=L, t_max=t_max)
        
        # Check against analytical at the final time step
        psi_exact = get_analytical(x, t_max, n_mode, c=c_wave, L=L)
        max_error = np.max(np.abs(psi_history[-1, :] - psi_exact))
        
        # Error should be reasonably small
        self.assertLess(max_error, 0.01, f"Wave solver error too high: {max_error}")


    # ==========================================
    # TESTS FOR DIFFUSION EQUATION
    # ==========================================

    def test_diffusion_boundaries(self):
        """Checks Dirichlet boundaries: top=1, bottom=0."""
        N = 20
        c_final = solve_diffusion_2d(N, T=0.01, dt=0.0001)
        
        self.assertTrue(np.allclose(c_final[0, :], 1.0), "Top boundary is not 1.0")
        self.assertTrue(np.allclose(c_final[-1, :], 0.0), "Bottom boundary is not 0.0")

    def test_diffusion_x_symmetry(self):
        """
        Since initial/boundary conditions only depend on y, 
        the solution should be identical across all x for a fixed y
        """
        N = 20
        c_final = solve_diffusion_2d(N, T=0.05, dt=0.0001)
        
        # For every row (y), the min and max value should be effectively identical
        for row in c_final:
            self.assertAlmostEqual(np.max(row), np.min(row), places=5, 
                                   msg="Symmetry broken in x-direction")

    def test_diffusion_steady_state_limit(self):
        """
        For t -> infinity, the concentration profile should be a straight line.
        """
        N = 10
        # Run for a long time (T=2.0) to approach steady state
        c_final = solve_diffusion_2d(N, T=2.0, dt=0.001)
        
        # Take a slice down the middle column
        profile = c_final[:, N//2]
        
        # Create the expected straight line from 1.0 to 0.0
        expected_profile = np.linspace(1.0, 0.0, N + 1)
        
        max_error = np.max(np.abs(profile - expected_profile))
        self.assertLess(max_error, 0.05, "Did not converge to straight line steady state")


    # ==========================================
    # TESTS FOR ITERATIVE SOLVERS (LAPLACE)
    # ==========================================

    
    def test_laplace_object_sink(self):
        """Checks if objects (sinks) correctly enforce c=0 in the domain."""
        from iterative_solvers import solve_laplace
        N = 20
        
        # Create a small 2x2 object in the center
        obj_mask = np.zeros((N + 1, N + 1), dtype=bool)
        obj_mask[10:12, 10:12] = True
        
        c_final, _ = solve_laplace(N, method='SOR', omega=1.5, object_mask=obj_mask)
        
        # 1. Assert the object itself is strictly 0.0
        self.assertTrue(np.all(c_final[obj_mask] == 0.0), "Object sink failed to maintain 0.0 concentration")
        
        # 2. Assert the surrounding area is NOT 0.0 (ensuring the whole grid didn't collapse)
        self.assertGreater(c_final[9, 10], 0.0, "Grid surrounding the sink erroneously collapsed to 0")
        self.assertGreater(c_final[12, 10], 0.0, "Grid surrounding the sink erroneously collapsed to 0")

if __name__ == '__main__':
    unittest.main()
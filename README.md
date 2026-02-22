# ScientificComputing
Scientific Computing Group 20

This README provides an overview of the implementation and execution of the numerical simulations for Assignment Set 1.

# Scientific Computing - Assignment 1 (Group 20)

**Authors:** Joeri Fresen (10607412), Alfonso Gondra (15843785), and Max van Beusekom (13979345).

This project involves the development and analysis of numerical solutions for the first assignment set of the scientific computing course.
## Project Structure

The code for this assignment is located in the `assignment_1/` directory.

### Core Solvers

* 
**`wave_equation.py`**: Implementation of the 1D wave equation solver using second-order central differences in space and time. It includes both standard explicit time-stepping and the symplectic Leapfrog integrator.


* 
**`diffusion.py`**: Implementation of the 2D time-dependent diffusion solver using an explicit forward-time central-space (FTCS) scheme and plots the results.


* 
**`iterative_solvers.py`**: Implementation of the Jacobi, Gauss-Seidel, and Successive Over-Relaxation (SOR) iterative methods for solving the steady-state Laplace equation.

## Requirements

The project is implemented in Python 3.10+ and requires the following libraries :

* `numpy`
* `matplotlib`
* `scipy`

## Usage

To reproduce the results and figures presented in the report, navigate to the `assignment_1` directory and execute the following scripts:

1. **Wave Equation Results:**
`experiment_wave.py`
This generates the combined wave development plots and the Leapfrog stability comparison in the `figures/` directory.

2. **Diffusion results**
`diffusion.py`
This generates the static and dynamic plots of the diffusion simulation and comparison against the analytical solution.

2. **Iterative Solver Performance:**
`experiment_convergence.py`
This script generates the log-lin convergence plots and the  optimization curves.

3. **Steady-State with Objects:**
`experiment_objects.py`
This produces the visualization of the "shadowing" effect caused by an absorbing sink in the domain.

## Results and Visualizations

Static plots and GIF animations of the time-dependent simulations (wave propagation and diffusion evolution) are stored in the `figures/` and `plots/` directories. The convergence measure  for iterative solvers is monitored until the tolerance  is reached.

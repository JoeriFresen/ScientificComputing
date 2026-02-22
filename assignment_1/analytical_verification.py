from scipy.special import erfc
import numpy as np

def analytical_diffusion_1d(y, t, D=1.0, n_terms=100):
    """Equation 9 from Assignment PDF """
    c = np.zeros_like(y)
    for i in range(n_terms):
        term1 = erfc((1 - y + 2*i) / (2 * np.sqrt(D * t)))
        term2 = erfc((1 + y + 2*i) / (2 * np.sqrt(D * t)))
        c += (term1 - term2)
    return c

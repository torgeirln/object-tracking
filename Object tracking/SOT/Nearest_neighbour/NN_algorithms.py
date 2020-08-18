import numpy as np
from scipy.stats import multivariate_normal

from Kalman_filters.kalman_filter import kalman_prediction, kalman_update
from Domain.Models.distributions import Gaussian

def NN_LG(p_prior, Z, PD, lamb_c, F, Q, H, R):
    """ 
        Nearest neighbour algoritm for linear and gaussian models 
        with constant probability of detection and uniform clutter.
    """
    # Predict
    x_k_kmin1, P_k_kmin1 = kalman_prediction(p_prior.x, p_prior.P, F, Q)
    
    # Compute w_tilde for all data associations theta
    mk = len(Z)
    w_tilde = np.zeros(mk+1)
    # - Precompute predicted likelihood
    z_bar, S = kalman_prediction(x_k_kmin1, P_k_kmin1, H, R)
    for theta in range(mk + 1):
        if theta == 0:
            w_tilde[theta] = 1 - PD
        else:
            z = Z[theta - 1]
            w_tilde[theta] = (PD / lamb_c) * multivariate_normal.pdf(z, mean=z_bar, cov=S)

    # Find most probable data association hypotesis
    theta_star = np.argmax(w_tilde)

    # Compute posterior based on theta_star
    x_k_k, P_k_k = x_k_kmin1, P_k_kmin1 # Assume theta_star is 0
    if theta_star > 0:
        z = Z[theta_star - 1]
        x_k_k, P_k_k = kalman_update(x_k_kmin1, P_k_kmin1, H, R, z)

    return Gaussian(x_k_k, P_k_k), Gaussian(x_k_kmin1, P_k_kmin1), theta_star

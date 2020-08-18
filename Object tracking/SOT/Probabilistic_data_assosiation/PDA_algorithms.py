import numpy as np
from scipy.stats import multivariate_normal

from Kalman_filters.kalman_filter import kalman_prediction, kalman_update
from Domain.Models.distributions import Gaussian


def PDA_LG(p_prior, Z, PD, lamb_c, F, Q, H, R):
    """ 
        Probabilistic data association filtering for linear and gaussian models 
        with constant probability of detection and uniform clutter.
    """
    # Dimensions
    nk = len(p_prior.x)
    mk = len(Z)

    # Predict
    x_k_kmin1, P_k_kmin1 = kalman_prediction(p_prior.x, p_prior.P, F, Q)

    # Update
    x_k_k = np.zeros((nk,mk+1))
    P_k_k = np.zeros((nk,nk,mk+1))
    w_tilde_theta = np.zeros(mk+1)
    # - Precompute predicted likelihood
    z_bar, S = kalman_prediction(x_k_kmin1, P_k_kmin1, H, R)
    for theta in range(mk + 1): 
        if theta == 0:
            # - Gaussian
            x_k_k[:,theta], P_k_k[:,:,theta] = x_k_kmin1, P_k_kmin1
            # - Weight
            w_tilde_theta[theta] = 1 - PD
        else:
            # - Gaussian
            z = Z[theta - 1]
            x_k_k[:,theta], P_k_k[:,:,theta] = kalman_update(x_k_kmin1, P_k_kmin1, H, R, z)
            # - Weight
            w_tilde_theta[theta] = (PD / lamb_c) * multivariate_normal.pdf(z, z_bar, S)
            
    # - Normalize weights
    w = w_tilde_theta / sum(w_tilde_theta)

    # Reduce to single gaussian density
    # - Mean
    x_PDA = np.zeros((nk,))
    for i in range(mk + 1):
        x_PDA += w[i] * x_k_k[:,i]
    # - Covariance
    P_PDA = np.zeros((nk,nk))
    for i in range(mk + 1):
        P_PDA += w[i] * P_k_k[:,:,i] + w[i] * (x_PDA - x_k_k[:,i]) @ (x_PDA - x_k_k[:,i]).T
    
    return Gaussian(x_PDA, P_PDA), Gaussian(x_k_kmin1, P_k_kmin1)

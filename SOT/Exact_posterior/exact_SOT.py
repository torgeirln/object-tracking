
import numpy as np
from scipy.stats import multivariate_normal

from Kalman_filters.kalman_filter import kalman_prediction, kalman_update
from Domain.Models.distributions import Gaussian, GaussianMixture
# from Domain.Models.distributions import WeightedGaussian, WeightedGaussianMixture

def exact_posterior_LG(p_prior, Z, PD, lamb_c, F, Q, H, R):
    """ 
        Exact posterior distribution given linear and gaussian models 
        with constant probability of detection and uniform clutter.
    """
    # Dimensions
    nk = p_prior.components[0].x.shape[0]
    mk = len(Z)
    # Allocate
    p_pred = []
    n_h_k = p_prior.n_components * (mk + 1)
    x_k_k = np.zeros((nk,n_h_k))
    P_k_k = np.zeros((nk,nk,n_h_k))
    w_tilde = np.zeros(n_h_k)
    h = -1 # Index for hypotesis at time k

    for p_h in p_prior.components:
        # Predict
        # - Gaussian, p_k|k-1(theta_1:k-1)
        x_k_kmin1, P_k_kmin1 = kalman_prediction(p_h.x, p_h.P, F, Q)
        # - Density, p(xk|Z_1:k-1)
        p_pred.append(
            Gaussian(x_k_kmin1, P_k_kmin1, p_h.w)
        )
        # Update
        # - Precompute predicted likelihood
        z_bar, S = kalman_prediction(x_k_kmin1, P_k_kmin1, H, R)
        # - Consider all data associations
        for theta in range(mk+1):
            # Update
            h += 1
            if theta == 0:
                # - Gaussian
                x_k_k[:,h], P_k_k[:,:,h] = x_k_kmin1, P_k_kmin1
                # - Weight 
                w_tilde[h] = p_h.w * (1 - PD)
            else:
                z = Z[theta-1]
                # - Gaussian
                x_k_k[:,h], P_k_k[:,:,h] = kalman_update(x_k_kmin1, P_k_kmin1, H, R, z)
                # - Weight
                w_tilde[h] = (p_h.w * PD / lamb_c) * multivariate_normal.pdf(z, mean=z_bar, cov=S)

    # Normalize weights
    w_k = w_tilde / sum(w_tilde)
    # Predicted and posterior densities
    p_pred = GaussianMixture(p_pred)
    p = GaussianMixture(
        [Gaussian(x_k_k[:,i], P_k_k[:,:,i], w_k[i]) for i in range(n_h_k)]
    )

    return p, p_pred


# def exact_posterior_LG_old(p_prior, Z, PD, lamb_c, F, Q, H, R):
#     """ 
#         Exact posterior distribution given linear and gaussian models 
#         with constant probability of detection and uniform clutter.
#     """
#     p_theta = []
#     p_pred = []

#     for w_prior, x_prior, P_prior in zip(*p_prior.moments):
#         # Predict
#         # - Gaussian, p_k|k-1(theta_1:k-1)
#         x_k_kmin1, P_k_kmin1 = kalman_prediction(x_prior, P_prior, F, Q)
#         # - Density, p(xk|Z_1:k-1)
#         p_pred.append(
#             WeightedGaussian(w_prior, x_k_kmin1, P_k_kmin1)
#         )

#         for theta in range(len(Z)+1):
#             # Update
#             if theta == 0:
#                 # - Weight 
#                 w_tilde_theta = w_prior * (1 - PD)
#                 # - Density
#                 p_theta.append(
#                     WeightedGaussian(w_tilde_theta, x_k_kmin1, P_k_kmin1)
#                 )
#             else:
#                 z = Z[theta-1]
#                 # - Gaussian
#                 x_k_k, P_k_k = kalman_update(x_k_kmin1, P_k_kmin1, H, R, z)
#                 # - Weight
#                 z_bar, S = kalman_prediction(x_k_kmin1, P_k_kmin1, H, R) # Predicted likelihood
#                 w_tilde_theta = (w_prior * PD / lamb_c) * multivariate_normal.pdf(z, mean=z_bar, cov=S)
#                 # - Density
#                 p_theta.append(
#                     WeightedGaussian(w_tilde_theta, x_k_k, P_k_k)
#                 )

#     p_pred = WeightedGaussianMixture(p_pred)
#     p = WeightedGaussianMixture(p_theta)
#     p.normalize_weights()
#     return p, p_pred

import numpy as np

from Domain.Models.distributions import Gaussian, GaussianMixture, MultiGaussianMixture
from Kalman_filters.kalman_filter import kalman_prediction, kalman_update

def HO_MHT_LG(p_prior, Z, n, PD, lamb_c, F, Q, H, R, Nmax, M, tol_prune):
    # Dimensions
    mk = len(Z)
    # Predict
    p_pred = []
    for p_h in p_prior.mixtures:
        p_pred_h = []
        for p_h_i in p_h.components:
            x_k_kmin1, P_k_kmin1 = kalman_prediction(p_h_i.x, p_h_i.P, F, Q)
            p_pred_h.append(
                Gaussian(x_k_kmin1, P_k_kmin1)
            )
        p_pred.append(
            GaussianMixture(p_pred_h, p_h.w)
        )
    p_pred = MultiGaussianMixture(p_pred, p_prior.ws)

    # Update
    # hk = 0 # Hypotesis index
    p = []
    for p_h_kmin1 in p_pred.mixtures:
        L = create_cost_matrix(p_h_kmin1, Z, n, PD, lamb_c, H, R)
        M = 10
        theta_stars = compute_M_associations(L, M)
        for m in range(theta_stars.shape[0]):
            p_h_k = []
            # hk += 1
            l_tilde = p_h_kmin1.w
            for i in range(n):
                if theta_stars[m,i] > mk: # Not associated
                    p_h_k.append(
                        Gaussian(p_h_kmin1.x, p_h_kmin1.P)
                    )
                else: # Associated
                    z = Z[theta_stars[m,i]]
                    x_k_k, P_k_k = kalman_update(p_h_kmin1.x, p_h_kmin1.P, H, R, z)
                    p_h_k.append(
                        Gaussian(x_k_k, P_k_k)
                    )
                l_tilde += L[i,theta_stars[m,i]]
            p.append(GaussianMixture(p_h_k, l_tilde)) # Yes but no
    
    # Normalize log-weights
    l = l_tilde / sum(l_tilde) # Yes but no
    p = MultiGaussianMixture(p, l)

            


def create_cost_matrix(p_h, Z, n, PD, lamb_c, H, R):
    mk = len(Z)
    # Allocate
    L_a = np.zeros((n,mk))
    L_ua = np.ones((n,n)) * np.inf
    L = np.hstack((L_a,L_ua))
    # Compute log weights
    for i, p_h_i in enumerate(p_h.components):
        # - Predicted likelihood
        z_h_i = H @ p_h_i.x
        S_h_i = H @ p_h_i.P @ H.T + R
        for theta in range(mk+1):
            # - Cost
            if theta == 0:
                L[i,mk+i] = -np.log(1 - PD)
            else:
                j = theta - 1
                z = Z[j]
                L[i,j] = -(np.log(PD/lamb_c) - 0.5 * np.log(np.linalg.det(2*np.pi*S_h_i)) - \
                        0.5 * (z - z_h_i).T @ np.linalg.inv(S_h_i) @ (z - z_h_i))
    return L

def compute_M_associations(L, M):
    return np.array([0])

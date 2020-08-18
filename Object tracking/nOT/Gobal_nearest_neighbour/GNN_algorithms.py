import numpy as np 
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment

from Domain.Models.distributions import Gaussian, GaussianMixture
from Kalman_filters.kalman_filter import kalman_prediction, kalman_update


def GNN_LG(p_prior, Z, n, PD, lamb_c, F, Q, H, R):
    """ 
        Global nearest neighbour algorithm for linear and gaussian models 
        with constant probability of detection and uniform clutter.
    """
    # Dimensions
    nk = p_prior.components[0].x.shape[0]
    mk = len(Z)
    # Allocate
    p_pred = []
    x_k_k = np.zeros((nk,n))
    P_k_k = np.zeros((nk,nk,n))
    # Predict
    for p_i in p_prior.components: # for each objects prior
        x_k_kmin1, P_k_kmin1 = kalman_prediction(p_i.x, p_i.P, F, Q)
        p_pred.append(
            Gaussian(x_k_kmin1, P_k_kmin1, p_i.w)
        )
    p_pred = GaussianMixture(p_pred)
    # Create cost matrix
    # - Allocate
    L_a = np.zeros((n,mk))
    L_ua = np.ones((n,n)) * np.inf
    L = np.hstack((L_a,L_ua))
    # - Compute log weights
    for i, p_i in enumerate(p_pred.components):
        # - Predicted likelihood
        z_h_i = H @ p_i.x
        S_h_i = H @ p_i.P @ H.T + R
        for theta in range(mk+1):
            # - Cost
            if theta == 0:
                L[i,mk+i] = -np.log(1 - PD)
            else:
                j = theta - 1
                z = Z[j]
                L[i,j] = -(np.log(PD/lamb_c) - 0.5 * np.log(np.linalg.det(2*np.pi*S_h_i)) - \
                        0.5 * (z - z_h_i).T @ np.linalg.inv(S_h_i) @ (z - z_h_i))
    # Find optimal assignment
    print(L)
    theta_star = optimal_assignment(L)
    # Compute posterior density
    for i in range(n):
        print(f'theta={theta_star[i]}')
        x_k_kmin1, P_k_kmin1 = p_pred.components[i].x, p_pred.components[i].P
        if theta_star[i] >= mk:
            x_k_k[:,i], P_k_k[:,:,i] = x_k_kmin1, P_k_kmin1
        else:
            z = Z[theta_star[i]]
            x_k_k[:,i], P_k_k[:,:,i] = kalman_update(x_k_kmin1, P_k_kmin1, H, R, z)
    p = GaussianMixture(
        [Gaussian(x_k_k[:,i], P_k_k[:,:,i]) for i in range(n)]
    )
    return p, p_pred, theta_star

def cost_matrix(w):
    # L_a = np.zeros((w.shape[0],w.shape[1]-1))
    # L_ua = np.zeros((w.shape[0],w.shape[0]))
    n = w.shape[0]
    L_a = -np.log(w[:,1:])
    print(L_a)
    L_ua = np.ones((n,n)) * np.inf
    l_ua = -np.log(w[:,0])
    L_ua[range(n),range(n)] = l_ua
    print(L_ua)
    L = np.hstack((L_a,L_ua))
    print(L)
    return L

def optimal_assignment(L):
    _, col_ind = linear_sum_assignment(L)
    return col_ind


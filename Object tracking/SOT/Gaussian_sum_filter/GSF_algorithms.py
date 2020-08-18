import numpy as np
from scipy.stats import multivariate_normal

from Kalman_filters.kalman_filter import kalman_prediction, kalman_update
from Domain.Models.distributions import Gaussian, GaussianMixture


def GSF_LG(p_prior, Z, PD, lamb_c, F, Q, H, R, Nmax, tol_prune=None, tol_merge=None):
    """ 
        Gaussian sum filtering for linear and gaussian models 
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
        x_k_kmin1, P_k_kmin1 = kalman_prediction(p_h.x, p_h.P, F, Q)
        p_pred.append(
            Gaussian(x_k_kmin1, P_k_kmin1, p_h.w)
        )
        # Update
        # - Precompute predicted likelihood
        z_bar, S = kalman_prediction(x_k_kmin1, P_k_kmin1, H, R)
        # - Consider all data associations
        for theta in range(mk+1):
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
                w_tilde[h] = (p_h.w * PD / lamb_c) * multivariate_normal.pdf(z, z_bar, S)

    # Normalize weights
    w_k = w_tilde / sum(w_tilde)

    # Reduction
    # - Pruning
    w_k, x_k_k, P_k_k = prune(w_k, x_k_k, P_k_k, tol_prune)
    # - Merging
    w_k, x_k_k, P_k_k = merge(w_k, x_k_k, P_k_k, tol_merge)
    # - Capping
    w_k, x_k_k, P_k_k = capp(w_k, x_k_k, P_k_k, Nmax)

    # Create final mixtures
    H_k = x_k_k.shape[1]
    p_pred = GaussianMixture(p_pred)
    p = GaussianMixture(
        [Gaussian(x_k_k[:,i], P_k_k[:,:,i], w_k[i]) for i in range(H_k)]
    )
    return p, p_pred
    

def prune(w, x, P, tol_prune):
    if tol_prune is not None:
        inds = w > tol_prune
        return w[inds], x[:,inds], P[:,:,inds]
    return w, x, P

def merge(w, x, P, tol_merge):
    if tol_merge is not None:
        print('WARNING: Merging is not yet implemented!')
        return w, x, P
    return w, x, P

def capp(w, x, P, Nmax):
    if len(w) > Nmax:
        n = x.shape[0]
        w_capp = np.zeros(Nmax)
        x_capp = np.zeros((n,Nmax))
        P_capp = np.zeros((n,n,Nmax))
        inds = np.argsort(-w) # Negate to get descending order
        for i in range(Nmax):
            w_capp[i] = w[inds[i]]
            x_capp[:,i] = x[:,inds[i]]
            P_capp[:,:,i] = P[:,:,inds[i]]
        return w_capp, x_capp, P_capp
    return w, x, P

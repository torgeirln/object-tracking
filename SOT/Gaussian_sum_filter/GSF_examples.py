import numpy as np
import matplotlib.pyplot as plt

from Domain.Models.clutter_models import UniformClutterModel1D
from Domain.Models.spaces import Space1D
from Domain.Models.distributions import Gaussian, GaussianMixture
from SOT.Gaussian_sum_filter.GSF_algorithms import GSF_LG
from SOT.Exact_posterior.exact_SOT import exact_posterior_LG
from UI.distribution_plots import plot_gaussian_mixture_pdf, plot_gaussian_pdf
from UI.measurements import plot_1D_measurements


def GSF_linear_gaussian_models_simple_scenario():
    arr = lambda scalar: np.array([[scalar]])
    # Prior
    x_prior = arr(0.5)
    P_prior = arr(0.2)
    p_prior_exact = GaussianMixture(Gaussian(x_prior, P_prior, weight=1))
    p_prior_GSF = GaussianMixture(Gaussian(x_prior, P_prior, weight=1))

    # Models
    # - Motion
    Q = arr(0.35)
    F = arr(1)
    # - Measurement
    R = arr(0.2)
    H = arr(1)
    PD = 0.9
    # - Clutter
    lamb = 0.4

    # GSF tuning params
    Nmax = 5
    tol_prune = 0.01 # or None

    # Sensor
    space = Space1D(-4,4)

    # Create measurement vector
    Z1 = [-1.3, 1.7]
    Z2 = [1.3]
    Z3 = [-0.3, 2.3]
    Z4 = [-2, 3]
    Z5 = [2.6]
    Z6 = [-3.5, 2.8]
    Zs = [Z1, Z2, Z3, Z4, Z5, Z6]
    
    # Plot settings
    ax = plt.subplot()
    res = 100
    marker_size = 100
    x_lim = [space.min, space.max]

    # Compute exact posterior
    for k, Z in enumerate(Zs, start=1):
        print(f'Measurements: {Z}')
        # Compute posterior
        p_exact, p_pred_exact = exact_posterior_LG(p_prior_exact, Z, PD, lamb, F, Q, H, R)
        p_GSF, p_pred_GSF = GSF_LG(p_prior_GSF, Z, PD, lamb, F, Q, H, R, Nmax, tol_prune)

        # Number of hypotesis
        print(f'Number of hypotesis, exact posterior: {p_exact.n_components}')
        print(f'Number of hypotesis, GSF posterior: {p_GSF.n_components}')

        # Plot
        ax.clear()
        ax.set(title=f'k = {k}', xlabel='x')
        ax.axhline(0, color='gray', linewidth=0.5)
        # - Predicted density according to GSF
        plt1 = plot_gaussian_mixture_pdf(ax, p_pred_GSF, x_lim, res=res, color='r', linestyle='--', zorder=0)
        # - Exact posterior density
        plt2 = plot_gaussian_mixture_pdf(ax, p_exact, x_lim, res=res, color='k', zorder=1)        
        # - Posterior density according to GSF
        plt3 = plot_gaussian_mixture_pdf(ax, p_GSF, x_lim, res=res, color='orange', linestyle='--', marker='x', zorder=2)
        # - Measurements
        plot_1D_measurements(ax, Z, color='b', marker='*', s=marker_size, zorder=3)
        # - Final details
        ax.set_xlim([space.min, space.max])
        ax.set_ylim([-0.05, 1.05])
        ax.legend((plt1, plt2, plt3), ('pred', 'exact', 'GSF'))
        plt.pause(0.0001)

        p_prior_exact = p_exact
        p_prior_GSF = p_GSF

        input('Press to continue')

def GSF_linear_gaussian_models_hard_scenario():
    arr = lambda scalar: np.array([[scalar]])
    # Prior
    x_prior = arr(0.5)
    P_prior = arr(0.2)
    p_prior_exact = GaussianMixture([Gaussian(x_prior, P_prior, weight=1)])
    p_prior_GSF = GaussianMixture([Gaussian(x_prior, P_prior, weight=1)])

    # Models
    # - Motion
    Q = arr(0.35)
    F = arr(1)
    # - Measurement
    R = arr(0.2)
    H = arr(1)
    PD = 0.9
    # - Clutter
    lamb = 0.4

    # GSF tuning params
    Nmax = 5
    tol_prune = 0.01 # or None

    # Sensor
    space = Space1D(-4,4)

    # Create measurement vector
    Z1 = [-1.3, 1.7]
    Z2 = [1.3]
    Z3 = [-0.3, 2.3]
    Z4 = [-0.7, 3]
    Z5 = [-1]
    Z6 = [-1.3]
    Zs = [Z1, Z2, Z3, Z4, Z5, Z6]
    
    # Plot settings
    ax = plt.subplot()
    res = 100
    marker_size = 100
    x_lim = [space.min, space.max]

    # Compute exact posterior
    for k, Z in enumerate(Zs, start=1):
        print(f'Measurements: {Z}')
        # Compute posterior
        p_exact, p_pred_exact = exact_posterior_LG(p_prior_exact, Z, PD, lamb, F, Q, H, R)
        p_GSF, p_pred_GSF = GSF_LG(p_prior_GSF, Z, PD, lamb, F, Q, H, R, Nmax, tol_prune)

        # Number of hypotesis
        print(f'Number of hypotesis, exact posterior: {p_exact.n_components}')
        print(f'Number of hypotesis, GSF posterior: {p_GSF.n_components}')

        # Plot
        ax.clear()
        ax.set(title=f'k = {k}', xlabel='x')
        ax.axhline(0, color='gray', linewidth=0.5)
        # - Predicted density according to GSF
        plt1 = plot_gaussian_mixture_pdf(ax, p_pred_GSF, x_lim, res=res, color='r', linestyle='--', zorder=0)
        # - Exact posterior density
        plt2 = plot_gaussian_mixture_pdf(ax, p_exact, x_lim, res=res, color='k', zorder=1)        
        # - Posterior density according to GSF
        plt3 = plot_gaussian_mixture_pdf(ax, p_GSF, x_lim, res=res, color='orange', linestyle='--', marker='x', zorder=2)
        # - Measurements
        plot_1D_measurements(ax, Z, color='b', marker='*', s=marker_size, zorder=3)
        # - Final details
        ax.set_xlim([space.min, space.max])
        ax.set_ylim([-0.05, 1.05])
        ax.legend((plt1, plt2, plt3), ('pred', 'exact', 'GSF'))
        plt.pause(0.0001)

        p_prior_exact = p_exact
        p_prior_GSF = p_GSF

        input('Press to continue')

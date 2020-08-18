import numpy as np
import matplotlib.pyplot as plt

from Domain.Models.clutter_models import UniformClutterModel1D
from Domain.Models.spaces import Space1D
from Domain.Models.distributions import Gaussian, GaussianMixture
from SOT.Exact_posterior.exact_SOT import exact_posterior_LG
from UI.distribution_plots import plot_gaussian_mixture_pdf


def exact_linear_gaussian_models_single_time_step():
    arr = lambda scalar: np.array([[scalar]])
    # Prior
    x_prior = arr(0.5)
    P_prior = arr(0.2)
    p_prior = Gaussian(x_prior, P_prior)
    # p_prior = GaussianMixture(Gaussian(x_prior, P_prior, weight=1))
    # p_prior = WeightedGaussian(weight=1, mean=x_prior, covariance=P_prior)

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

    # Sensor
    space = Space1D(-4,4)

    # Create measurement vector
    Z = [-1.3, 1.7]
    
    # Compute posterior
    p, p_pred = exact_posterior_LG(p_prior, Z, PD, lamb, F, Q, H, R)

    # Plot
    fig, axes = plt.subplots(4,1,figsize=(7,7))
    x_lim = [space.min, space.max]
    # - posterior density
    plot_gaussian_mixture_pdf(axes[0], p, x_lim, color='k', linewidth=3)
    # - predicted density
    plot_gaussian_mixture_pdf(axes[1], p_pred, x_lim, color='g', linewidth=3)
    # - prior density
    plot_gaussian_mixture_pdf(axes[2], p_prior, x_lim, color='g', linewidth=3)
    # - measurements and undetected hypotesis
    x_axis = np.zeros(len(Z))
    axes[3].scatter(Z, x_axis, color='b', marker='*', s=50)
    axes[3].set_xlim(x_lim)
    x_undetected = sum([gauss.w * gauss.x for gauss in p_pred.components])
    # x_undetected = sum([w * x for w, x in zip(p_pred.ws, p_pred.xs)])
    axes[3].scatter(x_undetected, 0, color='b', marker='o', s=50)

    plt.show()


def exact_linear_gaussian_models_several_time_steps():
    arr = lambda scalar: np.array([[scalar]])
    # Prior
    x_prior = arr(0.5)
    P_prior = arr(0.2)
    # p_prior = GaussianMixture([Gaussian(x_prior, P_prior, weight=1)])
    p_prior = Gaussian(x_prior, P_prior)
    # p_prior = WeightedGaussian(weight=1, mean=x_prior, covariance=P_prior)

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
    
    # Compute exact posterior
    for k, Z in enumerate(Zs, start=1):
        print(f'Measurements: {Z}')
        # Compute posterior
        p, p_pred = exact_posterior_LG(p_prior, Z, PD, lamb, F, Q, H, R)

        # Number of hypotesis
        print(f'number of hypotesis: {p.n_components}')

        # Plot
        fig, axes = plt.subplots(4,1,figsize=(7,7))
        x_lim = [space.min, space.max]
        # - posterior density
        plot_gaussian_mixture_pdf(axes[0], p, x_lim, color='k', linewidth=3)
        # p.plot(axes[0], lower=space.min, upper=space.max, color='k', linewidth=3)
        # - predicted density
        plot_gaussian_mixture_pdf(axes[1], p_pred, x_lim, color='g', linewidth=3)
        # p_pred.plot(axes[1], lower=space.min, upper=space.max, color='g', linewidth=3)
        # - prior density
        plot_gaussian_mixture_pdf(axes[2], p_prior, x_lim, color='g', linewidth=3)
        # p_prior.plot(axes[2], lower=space.min, upper=space.max, color='g', linewidth=3)
        # - measurements and undetected hypotesis
        x_axis = np.ones(len(Z)) * k
        axes[3].scatter(Z, x_axis, color='b', marker='*', s=50)
        axes[3].set_xlim(x_lim)
        x_undetected = sum([gauss.w * gauss.x for gauss in p_pred.components])
        # x_undetected = sum([w * x for w, x in zip(p_pred.ws, p_pred.xs)])
        axes[3].scatter(x_undetected, k, color='b', marker='o', s=50)

        p_prior = p

    plt.pause(0.0001)

import numpy as np
import matplotlib.pyplot as plt

from Domain.Models.distributions import Gaussian, GaussianMixture, MultiGaussianMixture
from Domain.Models.clutter_models import UniformClutterModel1D
from Domain.Models.spaces import Space1D
from nOT.Multi_hypotesis_tracker.MHT_algorithms import HO_MHT_LG
from Sensors.detectors import Detector1D
from UI.distribution_plots import plot_gaussian_mixture_pdf, plot_gaussian_pdf
from UI.measurements import plot_1D_measurements, plot_1D_heatmap


def MHT_LG_models_2_obj_1D():
    arr = lambda scalar: np.array([[scalar]])

    # Prior
    x_prior = arr(2.5)
    P_prior = arr(0.36)
    g_mix = GaussianMixture([Gaussian(-x_prior, P_prior), Gaussian(x_prior, P_prior)])
    n = 2
    # g_mix = GaussianMixture([Gaussian(-3, 0.2), Gaussian(3, 0.2)])
    p_prior = MultiGaussianMixture(g_mix)

    # Models
    # - Motion
    Q = arr(0.25)
    F = arr(1)
    # - Measurement
    R = arr(0.2)
    H = arr(1)
    PD = 0.85
    # - Clutter
    lamb = 0.3

    # Measurements
    Z = [-3.2, -2.4, 1.9, 2.2, 2.7, 2.9]
    Z = [arr(val) for val in Z]

    # Reduction
    Nmax = 100
    M = 10
    tol_prune = 0.1

    p, p_pred = HO_MHT_LG(p_prior, Z, n, PD, lamb, F, Q, H, R, Nmax, M, tol_prune)

    # Plot
    ax = plt.subplot()
    res = 100
    marker_size = 100
    x_lim = [-4, 4]
    colors=['orange', 'purple']
    ax.axhline(0, color='gray', linewidth=0.5)
    # - Measurements
    plot_1D_measurements(ax, Z, color='b', marker='*', s=marker_size, zorder=3)
    # - Object densities
    plts = []
    for i in range(n):
        # - Prior density
        plt1 = plot_gaussian_pdf(ax, p_prior.components[i], x_lim, res=res, color='g', zorder=1)        
        # - Predicted density
        plt2 = plot_gaussian_pdf(ax, p_pred.components[i], x_lim, res=res, color='r', linestyle='--', zorder=0)
        # - Posterior density
        plt3 = plot_gaussian_pdf(ax, p.components[i], x_lim, res=res, color=colors[i], linestyle='--', zorder=2)
        # - Association
        # if theta_star[i] >= len(Z):
        #     ax.scatter(p_pred.components[i].x, 0, color=colors[i], marker='o', s=marker_size, zorder=3)
        # else:
        #     ax.scatter(Z[theta_star[i]], 0, color=colors[i], marker='*', s=marker_size, zorder=3)
        [plts.append(plt) for plt in [plt1, plt2, plt3]]
    # - Final details
    # ax.set_xlim(x_lim)
    ax.set_ylim([-0.05, 1.5])
    ax.legend(plts, ('o1_prior', 'o1_pred', 'o1_GNN', 'o2_prior', 'o2_pred', 'o2_GNN'))
    plt.pause(0.0001)


def MHT_LG_models_2_obj_sequence_1D():
    arr = lambda scalar: np.array([[scalar]])

    # Space
    space = Space1D(-4,4)

    # Prior
    x_prior = arr(2.5)
    P_prior = arr(0.36)
    p_prior = GaussianMixture([Gaussian(-x_prior, P_prior), Gaussian(x_prior, P_prior)])

    # Models
    # - Motion
    Q = arr(0.25)
    F = arr(1)
    # - Measurement
    R = arr(0.2)
    H = arr(1)
    PD = 0.85
    _PD = lambda x : PD
    # - Clutter
    lamb = 0.4
    clutter_model = UniformClutterModel1D(space, lamb)

    # Create true sequence
    n = 2
    n_ks = 50
    X = np.vstack((-x_prior * np.ones(n_ks), x_prior * np.ones(n_ks)))

    # Measurements
    sensor = Detector1D(R, _PD, clutter_model, space)
    Z = sensor.get_measurements(X)

    # Estimate trajectories
    x_hat = np.zeros((n,n_ks))
    P_hat = np.zeros((n,n_ks))
    # thetas = np.zeros((n,n_ks))
    for k in range(n_ks):
        print(f'---k={k+1}---')
        print(Z[k])
        p, p_pred = HO_MHT_LG(p_prior, Z[k], n, PD, lamb, F, Q, H, R)
        for i in range(n):
            x_hat[i,k] = p.components[i].x
            P_hat[i,k] = p.components[i].P
        p_prior = p

    # Plot
    ax = plt.subplot()
    marker_size = 100
    colors = ['orange', 'purple']
    ass_colors = ['g', 'g']
    y_axis = np.linspace(1,n_ks,num=n_ks)
    # - Heatmaps
    for k in range(n_ks):
        for i in range(n):
            plot_1D_heatmap(ax, x_hat[i,k], P_hat[i,k], k+1)
    # - True trajectories
    for i in range(n):
        ax.plot(X[i,:], y_axis, color='k', marker='o')
    # - Measurements
    for k in range(n_ks):
        ax.axhline(k+1.5, color='gray', linewidth=0.5)
        plot_1D_measurements(ax, Z[k], k=k+1, color='b', marker='*', s=marker_size)
    # - Associations
    # for k in range(n_ks):
    #     for i in range(n):
    #         if thetas[i,k] < len(Z[k]):
    #             ax.scatter(Z[k][int(thetas[i,k])], k+1, color=ass_colors[i], marker='*', s=marker_size)
    # - Estimates
    for i in range(n):
        ax.plot(x_hat[i,:], y_axis, color=colors[i], marker='s')
    # - Final details
    ax.set(xlabel='x', ylabel='k')
    ax.set_xlim([space.min, space.max])

    plt.pause(0.0001)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

strd_y_lim = [-0.01,1]

def plot_gaussian_mixture_pdf(ax, gaussian_mixture, x_lim, y_lim=strd_y_lim, res=100, **kwargs):
    pdf = np.zeros(res)
    x_axis = np.linspace(x_lim[0], x_lim[1], res)
    for gaussian in gaussian_mixture.components:
        pdf += gaussian.w * multivariate_normal.pdf(x_axis, gaussian.x, gaussian.P)
    plt, = ax.plot(x_axis, pdf, **kwargs)
    ax.set_ylim(y_lim)
    return plt

def plot_gaussian_pdf(ax, gaussian, x_lim, y_lim=strd_y_lim, res=100, **kwargs):
    x_axis = np.linspace(x_lim[0], x_lim[1], res)
    pdf = gaussian.w * multivariate_normal.pdf(x_axis, gaussian.x, gaussian.P)
    plt, = ax.plot(x_axis, pdf, **kwargs)
    ax.set_ylim(y_lim)
    return plt

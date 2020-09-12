import numpy as np
from matplotlib import collections as matcoll
from scipy.stats import norm
from scipy.stats import multivariate_normal


def plot_1D_measurements(ax, Z, k=0, **kwargs):
    x_axis = np.ones(len(Z)) * k
    ax.scatter(Z, x_axis, **kwargs)

def plot_1D_heatmap(ax, x, P, y, **kwargs):
    res = 50
    a = 0.1
    sigma3 = 3 * np.sqrt(P)
    y_vals = np.ones(res) * y
    x_vals = np.linspace(x-sigma3,x+sigma3,num=res)
    alpha = a * multivariate_normal.pdf(x_vals, x, P)
    rgba_colors = np.zeros((res,4))
    rgba_colors[:,0] = 1.0
    rgba_colors[:,3] = alpha
    ax.scatter(x_vals, y_vals, color=rgba_colors, marker='s', s=100)


import numpy as np
from scipy.stats import multivariate_normal

class Gaussian():
    def __init__(self, mean, covariance, weight=1):
        self.x = mean
        self.P = covariance
        self.w = weight
        self.components = [self]
        self.n_components = 1

# class WeightedGaussian():
#     def __init__(self, weight, mean, covariance):
#         self.w = weight
#         self.x = mean
#         self.P = covariance
#         self.moments = ([weight], [mean], [covariance])
    
#     def plot(self, ax, lower, upper, res=100, **kwargs):
#         x_axis = np.linspace(lower, upper, res)
#         ax.plot(x_axis, self.pdf(x_axis), **kwargs)
#         ax.set_ylim([-0.01,1])
    
#     def pdf(self, linspace, res=100):
#         return self.w * multivariate_normal.pdf(linspace, self.x, self.P,)
    

# class WeightedGaussianMixture():
#     def __init__(self, weighted_gaussians):
#         self.ws = [wg.w for wg in weighted_gaussians] 
#         self.xs = [wg.x for wg in weighted_gaussians]
#         self.Ps = [wg.P for wg in weighted_gaussians]
#         self.moments = (self.ws, self.xs, self.Ps)
#         self.n_components = len(weighted_gaussians)

#     def normalize_weights(self):
#         self.ws = self.ws / sum(self.ws)
#         self.moments = (self.ws, self.xs, self.Ps)
    
#     def plot(self, ax, lower, upper, res=100, **kwargs):
#         pdf = np.zeros(res)
#         x_axis = np.linspace(lower, upper, res)
#         for w, x, P in zip(*self.moments):
#             pdf += WeightedGaussian(w, x, P).pdf(x_axis)
#         plt, = ax.plot(x_axis, pdf, **kwargs)
#         ax.set_ylim([-0.01,1])
#         return plt

class GaussianMixture():
    def __init__(self, gaussians, weight=1):
        if type(gaussians) is Gaussian:
            self.components = [gaussians]
        else:
            self.components = gaussians
        self.w = weight
        self.n_components = len(self.components)

class MultiGaussianMixture():
    def __init__(self, gaussian_mixtures, weights=None):
        if type(gaussian_mixtures) is GaussianMixture:
            self.mixtures = [gaussian_mixtures]
        else:
            self.mixtures = gaussian_mixtures
        self.ws = weights if weights is not None else [1 for _ in range(len(self.mixtures))]
        self.n_mixtures = len(self.mixtures)

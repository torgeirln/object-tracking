import sys
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from SOT.Sensors.detectors import Detector1D, Detector2D
from Domain.Models.clutter_models import UniformClutterModel1D, UniformClutterModel2D
from Domain.Models.spaces import Space1D, Space2D

def test_2D_detector_w_uniform_clutter():
    done = False
    ax = plt.subplot(aspect='equal')
    while not done:
        space = np.array([[-4,4],[-4,4]])
        space = Space2D(-4, 4, -4, 4)
        lamb = 0.1
        clutter_model = UniformClutterModel2D(space, lamb)

        R = np.array([[0.3,0],[0,0.3]])
        PD = lambda x : 0.85
        sensor = Detector2D(R, PD, clutter_model, space)

        x = np.linspace([0,0], [2,2])
        x = x[15,:]
        x = np.array([3,2])
        print(f'x = {x}')
        Z = sensor.get_single_measurement(x)

        ax.clear()
        plot_2Dmeasurements(ax, Z)
        plot_sigma_ellipse(ax, mu=x, S=R, level=3, color='g', linestyle=':')
        space.plot(ax, linestyle=':', color='gray')

        plt.pause(0.00001)

        done = input('Press q to quit. ') == 'q'

def test_1D_detector_w_uniform_clutter():
    done = False
    ax = plt.subplot()
    while not done:
        space = Space1D(-4, 4)
        lamb = 0.1
        clutter_model = UniformClutterModel1D(space, lamb)

        R = 0.3
        PD = lambda x : 0.85
        sensor = Detector1D(R, PD, clutter_model, space)

        x = 0
        print(f'x = {x}')
        Z = sensor.get_single_measurement(x)

        ax.clear()
        plot_1Dmeasurements(ax, Z)
        space.plot(ax, linestyle=':', color='gray')

        plt.pause(0.00001)

        done = input('Press q to quit. ') == 'q'

def plot_2Dmeasurements(ax, Z):
    if len(Z) >= 1:
        ax.scatter(Z[:,0], Z[:,1], color='r', marker='x')
        
def plot_1Dmeasurements(ax, Z):
    if len(Z) >= 1:
        ax.scatter(Z[:], np.zeros((len(Z))), color='r', marker='x')

def plot_sigma_ellipse(ax, mu, S, level=3, n_points=50, **kwargs):
    # Evenly spaced vector of angles
    phi = np.linspace(0, 2*np.pi, n_points)
    # Vector of circle points
    z = level * np.array([np.cos(phi), np.sin(phi)])
    # 2D points according to equation (2)
    xy = mu.reshape(-1,1) + sqrtm(S) @ z
    # Plot
    ax.plot(xy[0,:], xy[1,:], **kwargs)

if __name__ == "__main__":
    # test_2D_detector_w_uniform_clutter()
    
    test_1D_detector_w_uniform_clutter()

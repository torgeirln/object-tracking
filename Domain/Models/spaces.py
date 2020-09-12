
import numpy as np


class Space1D():
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def contains(self, z):
        cond1 = z >= self.min
        cond2 = z <= self.max
        return cond1 & cond2
    
    def plot(self, ax, padding=1, **kwargs):
        ax.axvline(self.min, **kwargs)
        ax.axvline(self.max, **kwargs)
        ax.set_xlim([self.min - padding, self.max + padding])


class Space2D():
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def contains(self, z):
        cond1 = z[0] >= self.x_min
        cond2 = z[0] <= self.x_max
        cond3 = z[1] >= self.y_min
        cond4 = z[1] <= self.y_max
        return cond1 & cond2 & cond3 & cond4
        
    def plot(self, ax, padding=1, **kwargs):
        left = np.array([[self.x_min, self.x_min], [self.y_min, self.y_max]])
        right = np.array([[self.x_max, self.x_max], [self.y_min, self.y_max]])
        top = np.array([[self.x_min, self.x_max], [self.y_max, self.y_max]])
        bottom = np.array([[self.x_min, self.x_max], [self.y_min, self.y_min]])

        ax.plot(left[0,:], left[1,:], **kwargs)
        ax.plot(right[0,:], right[1,:], **kwargs)
        ax.plot(top[0,:], top[1,:], **kwargs)
        ax.plot(bottom[0,:], bottom[1,:], **kwargs)

        ax.set_xlim([self.x_min - padding, self.x_max + padding])
        ax.set_ylim([self.y_min - padding, self.y_max + padding])

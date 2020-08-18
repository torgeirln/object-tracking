
import numpy as np


class UniformClutterModel1D():
    def __init__(self, space, lamb):
        self.space = space
        self.lamb = lamb
        self.V = space.max - space.min
        self.lamb_bar_c = self.lamb * self.V
        
    def lamb_c(self, c):
        if self.space.contains(c):
            return self.lamb
        else:
            return 0

    def f_c(self, c):
        if self.space.contains(c):
            return 1 / self.V
        else:
            return 0

    def get_clutter(self):
        C = []
        m_clutter = np.random.poisson(self.lamb_bar_c)
        # print(f'lamb_bar_c = {self.lamb_bar_c}')
        # print(f'm_clutter = {m_clutter}')
        for _ in range(m_clutter):
            c = np.random.uniform(low=self.space.min, high=self.space.max)
            C.append(np.array([[c]]))
        return C


class UniformClutterModel2D():
    def __init__(self, space, lamb):
        self.space = space
        self.lamb = lamb
        self.V = (space.x_max - space.x_min) * (space.y_max - space.y_min)
        self.lamb_bar_c = self.lamb * self.V
        
    def lamb_c(self, c):
        if self.space.contains(c): 
            return self.lamb
        else:
            return 0

    def f_c(self, c):
        if self.space.contains(c):
            return 1 / self.V
        else:
            return 0

    def get_clutter(self):
        C = []
        m_clutter = np.random.poisson(self.lamb_bar_c)
        print(f'lamb_bar_c = {self.lamb_bar_c}')
        print(f'm_clutter = {m_clutter}')
        for _ in range(m_clutter):
            cx = np.random.uniform(low=self.space.x_min, high=self.space.x_max)
            cy = np.random.uniform(low=self.space.y_min, high=self.space.y_max)
            c = np.array([cx, cy])
            C.append(c)
        return C

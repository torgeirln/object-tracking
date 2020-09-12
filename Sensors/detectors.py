
import numpy as np


class Detector1D():
    def __init__(self, variance, detection_prob, clutter_model, space):
        self.R = variance
        self.PD = detection_prob
        self.C_model = clutter_model
        self.space = space

    def get_single_measurement(self, x):
        z = []
        o = self.get_object_sample(x)
        if o is not None:
            z.append(o)
        # r = np.random.uniform()
        # if r <= self.PD(x):
        #     o = self.get_object_sample(x)
        #     print(f'o = {o}')
        #     z.append(o)
        C = self.C_model.get_clutter()
        [z.append(c) for c in C]
        return np.array(z)

    def get_measurements(self, X):
        Z = []
        n = X.shape[0] # Number of objects
        for k in range(X.shape[1]):
            Zk = []
            for i in range(n):
                o = self.get_object_sample(X[i,k])
                if o is not None:
                    Zk.append(o)
            C = self.C_model.get_clutter()
            [Zk.append(c) for c in C]
            # Z.append(np.array(Zk))
            Z.append(Zk)
        return Z

    def get_object_sample(self, x):
        r = np.random.uniform()
        if r <= self.PD(x):
            o_in_fov = False
            o = None
            # If x is close to the edge o might be sampled outside the fov
            while not o_in_fov:
                o = np.random.normal(loc=x, scale=np.sqrt(self.R))
                o_in_fov = self.space.contains(o)
            return o
        return None
        

class Detector2D():
    def __init__(self, noise_covar, detection_prob, clutter_model, space):
        """ Input: 
                noise_covar - numpy array (matrix)
                detection_prob - float
                clutter_intensity - float                
        """ 
        self.R = noise_covar
        self.PD = detection_prob
        self.C_model = clutter_model
        self.space = space

    def get_single_measurement(self, x):
        z = []
        r = np.random.uniform()
        if r <= self.PD(x):
            o = self.get_object_sample(x)
            print(f'o = {o}')
            z.append(o)
        C = self.C_model.get_clutter()
        [z.append(c) for c in C]
        return np.array(z)

    def get_object_sample(self, x):
        o_in_fov = False
        o = None
        # If x is close to the edge o might be sampled outside the fov
        while not o_in_fov:
            o = np.random.multivariate_normal(mean=x, cov=self.R)
            o_in_fov = self.space.contains(o)
        return o

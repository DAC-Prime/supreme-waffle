import numpy as np 

# RescaleNormalizer
class RescaleNormalizer:
    def __init__(self, coef = 1.0):
        self.coef = coef

    def __call__(self, x):
        x = np.asarray(x)
        return self.coef * x
import numpy as np
from .linear_regression import Linear_Regression
from itertools import combinations_with_replacement

class Polynomial_Regression(Linear_Regression):
    def _trans(self, x):
        coms = []
        for d in range(1, self.degree + 1):
            coms.extend(combinations_with_replacement(range(x.shape[1]), d))
        poly = np.empty((x.shape[0], len(coms)), dtype=x.dtype)
        for i, comp in enumerate(coms):
            if len(comp) == 1: poly[:, i] = x[:, comp[0]]
            else: poly[:, i] = np.prod(x[:, comp], axis=1)
        return poly

    def fit(self, x, y, degree=1, **kwargs):
        self.degree = degree
        poly = self._trans(x)
        super().fit(poly, y, **kwargs)

    def predict(self, x):
        poly = self._trans(x)
        return super().predict(poly)
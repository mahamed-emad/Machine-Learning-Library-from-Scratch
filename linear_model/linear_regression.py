import numpy as np
from optimizers import *
from regularizers import *
from .linear import Linear

class Linear_Regression(Linear):
    def fit(self, x, y, optimizer=None, regularizer=None):
        if optimizer is None: optimizer = Optimizer()
        if regularizer is None: regularizer = Regularizer()

        size = (x.shape[0], x.shape[1] + 1)
        self.weights = np.zeros((size[1], 1))
        x = np.hstack((np.full((size[0], 1), 1), x))

        self.batches = self.batches if self.batches else size[0]
        num_batch = (size[0]-1)//self.batches + 1
        for _ in range(self.itr):
            idx = np.random.permutation(size[0])
            x, y = x[idx], y[idx]
            for i in range(num_batch):
                _x, _y = x[self.batches*i: self.batches*(i + 1)], y[self.batches*i: self.batches*(i + 1)]
                y_hat = np.dot(_x, self.weights)
                error = y_hat - _y
                dw_mse = np.dot(_x.T, error) / error.shape[0]
                dw_reg = regularizer.grad(self.weights)
                dw = dw_reg + dw_mse
                dw = optimizer.update(dw)
                self.weights -= self.lr * dw
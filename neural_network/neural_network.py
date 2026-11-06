from .mlp import *
import numpy as np
from .activations import *
from .losses import *

class Neural_Network:
    def __init__(self, dimens=[], acts_out=None, niter=int(1e4), lr=0.1, hide_func=Relu, loss_fn=None):
        self.lr = lr
        self.niter = niter
        self.dimens = dimens        
        self.acts_out = acts_out() if acts_out else Activation()
        self.hide_func = hide_func if hide_func else Activation
        self.loss_fn, self.clas = (loss_fn(), True) if loss_fn else (MSE(), False)
    
    def colm_x(self, x):
        x = np.array(x, dtype=float).T
        if x.ndim == 1: x = x.reshape(1, -1)
        x = np.vstack((np.full((1, x.shape[1]), 1), x))
        return np.array(x)
    
    def fit(self, x, y):
        x = self.colm_x(x)

        self.x_mean, self.x_std = x.mean(), x.std()
        self.x_std = 1e-9 if not self.x_std else self.x_std
        x = (x - self.x_mean) / (self.x_std)
        self.y_mean, self.y_std = y.mean(), y.std()
        self.y_std = 1e-9 if not self.y_std else self.y_std
        if self.clas: self.y_mean, self.y_std = 0.0, 1.0
        y = (y - self.y_mean) / self.y_std

        self.dimens[0] = x.shape[0]
        self.propagation = Propagation(x, y, self.dimens, self.acts_out, self.lr, self.hide_func, self.loss_fn)
        for _ in range(int(self.niter)):
            self.propagation.forward(x)
            self.propagation.backward()
    
    def predict(self, x):
        x = self.colm_x(x)
        x = (x - self.x_mean) / (self.x_std)
        return self.propagation.forward(x).ravel() * self.y_std + self.y_mean
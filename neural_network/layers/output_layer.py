import numpy as np
from .layer import *

class Out_Layer(Layer):
    def __init__(self, y, lr, num_sample, num_node, func):
        self.lr = lr
        y = np.array(y)
        self.func = func
        self.prev = self.next = None
        self.y = y.reshape(1, -1) if y.ndim == 1 else y

    def forward(self):
        self.result_prev = self.prev.result
        self.result = self.func.calc(self.result_prev)
        return self.result

    def backward(self):
        error = self.result - self.y
        d_actv = self.func.derv(self.result_prev)
        self.derv = error * d_actv
        self.derv = np.vstack((np.full((1, self.derv.shape[1]), 1), self.derv))
        self.prev.backward()
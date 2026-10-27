import numpy as np
from .layer import *

class Hiden_Layer(Layer):
    def __init__(self, lr, num_sample, num_node, num_next_node, hide_func):
        self.lr = lr
        self.hide_func = hide_func()
        self.num_node = num_node
        self.prev = self.next = None
        self.num_sample = num_sample
        self.num_next_node = num_next_node
        self.weights = np.random.randn(self.num_node, self.num_next_node) * 0.1
    
    def forward(self):
        self.result_prev = np.vstack((np.full((1, self.prev.result.shape[1]), 1), self.prev.result))
        self.func = self.hide_func.calc(self.result_prev)
        self.result = np.dot(self.weights.T, self.func)
        return self.next.forward()
    
    def backward(self):
        delta = self.next.derv[1:, :]
        self.derv = np.dot(self.weights, delta)
        self.derv = self.derv * self.hide_func.derv(self.result_prev)
        dw = np.dot(self.func, delta.T)
        self.weights = self.weights - ((self.lr * dw) / self.num_sample)
        self.prev.backward()
import numpy as np
from .layer import *

class Input_Layer(Layer):
    def __init__(self, x, lr, num_next_node):
        self.lr = lr
        self.num_node = x.shape[0]
        self.prev = self.next = None
        self.num_next_node = num_next_node
        self.weights = np.random.randn(self.num_node, self.num_next_node) * 0.1
        
    def forward(self, x):
        self.x = x
        self.result = np.dot(self.weights.T, x)
        return self.next.forward()
        
    def backward(self):
        delta = self.next.derv[1:, :]
        self.derv = np.dot(self.x, delta.T)
        self.weights = self.weights - ((self.lr * self.derv) / self.x.shape[1])
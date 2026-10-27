import numpy as np
from .layers import *

class Propagation:
    def __init__(self, x, y, dimens, acts_out, lr, hide_func):
        self.lr = lr
        self.dimens = [i + 1 for i in dimens]
        self.dimens[-1] -= 1
        self.acts_out = acts_out
        self.feature, self.num_sample = x.shape
        self.head = prev = Input_Layer(x, self.lr, self.dimens[1] -1)
        for i in range(1, len(dimens) -1):
            now = Hiden_Layer(self.lr, self.num_sample, self.dimens[i], max(1, self.dimens[i+1] -1), hide_func)
            prev.next = now
            now.prev = prev
            prev = now
        self.tail = now = Out_Layer(y, self.lr, self.num_sample, self.dimens[-2], acts_out)
        prev.next = now
        now.prev = prev

    def forward(self, x):
        return self.head.forward(x)

    def backward(self):
        self.tail.backward()
import numpy as np
from ._base import Base

class Adam(Base):
    def __init__(self):
        self.t = 0
        self.mw = None
        self.mv = None

    def update(self, dw):
        if self.mw is None:
            self.mw = np.zeros_like(dw)
            self.mv = np.zeros_like(dw)
        self.t += 1
        self.mw = 0.9 * self.mw + 0.1 * dw
        self.mv = 0.999 * self.mv + 0.001 * (dw ** 2)
        _mw = self.mw / (1 - 0.9 ** self.t)
        _mv = self.mv / (1 - 0.999 ** self.t)
        return _mw / (_mv ** 0.5 + 1e-9)
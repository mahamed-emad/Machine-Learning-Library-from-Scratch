import numpy as np
from ._base import Base

class Ridge(Base):
    def __init__(self, alphta=0.01):
        self.alphta = alphta

    def grad(self, weights):
        result = weights.copy()
        result[0] = 0
        return self.alphta * result
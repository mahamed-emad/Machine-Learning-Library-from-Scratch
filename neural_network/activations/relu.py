from ._base import *

class Relu(Base):
    def calc(self, result):
        return np.maximum(result, 0)
    
    def derv(self, result):
        return np.where(result > 0, 1, 0)
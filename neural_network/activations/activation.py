from ._base import *

class Activation(Base):
    def calc(self, result):
        return result
    
    def derv(self, result):
        return np.full_like(result, 1)
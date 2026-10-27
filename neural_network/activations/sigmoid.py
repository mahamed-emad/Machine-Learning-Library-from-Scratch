from ._base import *

class Sigmoid(Base):
    def calc(self, result):
        result = np.clip(result, -500, 500)
        return ((np.e ** (-1 * result)) + 1) ** -1
    
    def derv(self, result):
        return (self.calc(result) * (1 - self.calc(result)))
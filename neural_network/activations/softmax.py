from ._base import *

class Softmax(Base):
    def calc(self, result):
        result = result - np.max(result, axis=0, keepdims=True)
        result = np.e ** result
        result = result / np.sum(result, axis=0, keepdims=True)
        return result
    
    def derv(self, result):
        return np.ones_like(result)
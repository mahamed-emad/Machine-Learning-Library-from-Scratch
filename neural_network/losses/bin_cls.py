from ._base import *

class Bin_Cls(Base):
    def derv(self, y_pred, y, func, result_prev):
        return y_pred - y
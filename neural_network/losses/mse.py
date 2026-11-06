from ._base import *

class MSE(Base):
    def derv(self, y_pred, y, func, result_prev):
        error = (y_pred - y)
        d_actv = func.derv(result_prev)
        return error * d_actv
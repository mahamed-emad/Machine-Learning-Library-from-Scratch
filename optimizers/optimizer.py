from ._base import Base

class Optimizer(Base):
    def update(self, dw):
        return dw
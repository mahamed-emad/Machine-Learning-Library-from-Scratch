from ._base import Base

class Regularizer(Base):
    def __init__(self, alphta=0.01):
        self.alphta = alphta

    def grad(self, weight):
        return 0
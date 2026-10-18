import numpy as np
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def grad(self):
        pass
import numpy as np
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def derv(self): pass
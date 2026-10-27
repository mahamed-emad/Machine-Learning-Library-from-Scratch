import numpy as np
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def calc(self, result):
        pass
    
    @abstractmethod
    def derv(self, result):
        pass
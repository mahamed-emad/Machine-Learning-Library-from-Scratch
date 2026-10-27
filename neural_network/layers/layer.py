from ..activations import *
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def backward(self):
        pass
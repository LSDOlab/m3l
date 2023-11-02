from m3l.utils.parameters import Parameters
from abc import ABC, abstractmethod



class OperationBase(ABC):

    def __init__(self, **kwargs) -> None:
    
        self.parameters = Parameters()
        self.initialize(kwargs)
        self.parameters.update(kwargs)

    @abstractmethod
    def initialize(self, kwargs):
        raise NotImplementedError
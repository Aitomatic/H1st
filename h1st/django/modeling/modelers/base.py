from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..models import Model


__all__: Sequence[str] = 'Modeler', 'H1stModeler'


class Modeler(ABC):
    @property
    @abstractmethod
    def native_model_class(self):
        raise NotImplementedError

    @abstractmethod
    def create_model(self) -> Model:
        raise NotImplementedError


# alias
H1stModeler = Modeler

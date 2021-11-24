from abc import ABC, abstractmethod
from typing import Sequence   # TODO: Py3.9: use generic collections.abc

from ..models import Model


__all__: Sequence[str] = 'Modeler', 'H1stModeler'


class Modeler(ABC):
    @property
    @abstractmethod
    def native_model_class(self):
        raise NotImplementedError

    @abstractmethod
    def create_model(self, *args, **kwargs) -> Model:
        raise NotImplementedError


# alias
H1stModeler = Modeler

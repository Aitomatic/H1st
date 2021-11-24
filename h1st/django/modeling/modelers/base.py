from abc import abstractmethod
import sys

from ..models import Model

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = 'Modeler', 'H1stModeler'


class Modeler:
    @property
    @abstractmethod
    def native_model_class(self):
        raise NotImplementedError

    @abstractmethod
    def create_model(self, *args, **kwargs) -> Model:
        raise NotImplementedError


# alias
H1stModeler = Modeler

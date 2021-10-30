from typing import Sequence   # TODO: Py3.9: use generic collections.abc

from torch.serialization import load

from .....util import PGSQL_IDENTIFIER_MAX_LEN
from ....apps import H1stAIModelingModuleConfig
from ..base import H1stMLModel


__all__: Sequence[str] = 'TorchModel', 'H1stTorchModel'


class TorchModel(H1stMLModel):
    class Meta(H1stMLModel.Meta):
        verbose_name: str = 'H1st Torch Model'
        verbose_name_plural: str = 'H1st Torch Models'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_torch_models'

    def load(self):
        self.native_model_obj = load(f=self.artifact_local_path)


# alias
H1stTorchModel = TorchModel

from .....util import PGSQL_IDENTIFIER_MAX_LEN
from ....apps import H1stAIModelingModuleConfig
from ..base import H1stMLModel


class TorchModel(H1stMLModel):
    class Meta(H1stMLModel.Meta):
        verbose_name = 'H1st Torch Model'
        verbose_name_plural = 'H1st Torch Models'

        db_table = (f'{H1stAIModelingModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_torch_models'


# alias
H1stTorchModel = TorchModel

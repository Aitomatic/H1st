from collections.abc import Sequence

from .....util import PGSQL_IDENTIFIER_MAX_LEN
from ....apps import H1stAIModelingModuleConfig
from ..base import H1stMLModel


__all__: Sequence[str] = 'TFModel', 'H1stTFModel'


class TFModel(H1stMLModel):
    class Meta(H1stMLModel.Meta):
        verbose_name: str = 'H1st TensorFlow Model'
        verbose_name_plural: str = 'H1st TensorFlow Models'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_tf_models'


# alias
H1stTFModel = TFModel

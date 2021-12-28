from typing import Sequence   # TODO: Py3.9: use generic collections.abc

from .....util import PGSQL_IDENTIFIER_MAX_LEN
from ....apps import H1stAIModelingModuleConfig

from ..base import PyLoadablePreTrainedMLModel


__all__: Sequence[str] = ('PreTrainedHuggingFaceTransformer',
                          'H1stPreTrainedHuggingFaceTransformer')


class PreTrainedHuggingFaceTransformer(PyLoadablePreTrainedMLModel):
    class Meta(PyLoadablePreTrainedMLModel.Meta):
        verbose_name: str = 'Pre-Trained Hugging Face Transformer'
        verbose_name_plural: str = 'Pre-Trained Hugging Face Transformers'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_pretrained_hugging_face_transformers'


# alias
H1stPreTrainedHuggingFaceTransformer = PreTrainedHuggingFaceTransformer

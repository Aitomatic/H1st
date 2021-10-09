__all__ = 'PreTrainedTFHubTransformer', 'H1stPreTrainedTFHubTransformer'


from ......util import PGSQL_IDENTIFIER_MAX_LEN
from .....apps import H1stAIModelingModuleConfig

from ...base import PyLoadablePreTrainedMLModel


class PreTrainedTFHubTransformer(PyLoadablePreTrainedMLModel):
    class Meta(PyLoadablePreTrainedMLModel.Meta):
        verbose_name = 'H1st Pre-Trained TensorFlow Hub Transformer'
        verbose_name_plural = 'H1st Pre-Trained TensorFlow Hub Transformers'

        db_table = (f'{H1stAIModelingModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_pretrained_tf_hub_transformers'


# alias
H1stPreTrainedTFHubTransformer = PreTrainedTFHubTransformer

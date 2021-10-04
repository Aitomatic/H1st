from tensorflow.python.keras.saving.save import load_model

from .....util import PGSQL_IDENTIFIER_MAX_LEN
from ....apps import H1stModelModuleConfig
from ..base import H1stMLModel


class KerasModel(H1stMLModel):
    class Meta(H1stMLModel.Meta):
        verbose_name = 'H1st Keras Model'
        verbose_name_plural = 'H1st Keras Models'

        db_table = \
            f"{H1stModelModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_keras_models'

    def load(self):
        self._native_obj = load_model(filepath=self.artifact_local_path,
                                      custom_objects=None,
                                      compile=True,
                                      options=None)


# alias
H1stKerasModel = KerasModel

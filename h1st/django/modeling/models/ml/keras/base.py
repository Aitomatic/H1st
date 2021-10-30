from typing import Sequence   # TODO: Py3.9: use generic collections.abc
from pathlib import Path

import h5py
from tensorflow.python.keras.saving.save import load_model

from .....util import PGSQL_IDENTIFIER_MAX_LEN
from ....apps import H1stAIModelingModuleConfig
from ..base import H1stMLModel


__all__: Sequence[str] = 'KerasModel', 'H1stKerasModel'


class KerasModel(H1stMLModel):
    class Meta(H1stMLModel.Meta):
        verbose_name: str = 'H1st Keras Model'
        verbose_name_plural: str = 'H1st Keras Models'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_keras_models'

    def load(self) -> None:
        artifact_local_path = str(Path(self.artifact_local_path).expanduser())

        self.native_model_obj = \
            load_model(filepath=(h5py.File(name=artifact_local_path, mode='r')
                                 if artifact_local_path.endswith('.h5')
                                 else artifact_local_path),
                       custom_objects=None,
                       compile=True,
                       options=None)


# alias
H1stKerasModel = KerasModel

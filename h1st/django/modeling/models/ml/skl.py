from typing import Sequence   # TODO: Py3.9: use generic collections.abc

import joblib

from ....util import PGSQL_IDENTIFIER_MAX_LEN
from ...apps import H1stAIModelingModuleConfig
from .base import H1stMLModel


__all__: Sequence[str] = 'SKLModel', 'H1stSKLModel'


class SKLModel(H1stMLModel):
    class Meta(H1stMLModel.Meta):
        verbose_name: str = 'H1st SciKit-Learn Model'
        verbose_name_plural: str = 'H1st SciKit-Learn Models'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_skl_models'

    # by default, serialize model object by JobLib/Pickle
    def save(self, *args, **kwargs):
        assert self.native_model_obj, \
            ValueError(f'*** MODEL OBJECT {self.native_model_obj} INVALID ***')

        joblib.dump(value=self.native_model_obj,
                    filename=self.artifact_local_path,
                    compress=0,
                    protocol=3,   # default protocol in Python 3.0–3.7
                    cache_size=None)

        super().save(*args, **kwargs)

    # by default, deserialize model object by JobLib/Pickle
    def load(self) -> None:
        if self.native_model_obj is None:
            self.native_model_obj = \
                joblib.load(filename=self.artifact_local_path, mmap_mode=None)


# alias
H1stSKLModel = SKLModel

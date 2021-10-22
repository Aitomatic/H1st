from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Optional

from django.db.models.fields import CharField

from ....util import PGSQL_IDENTIFIER_MAX_LEN, import_obj
from ...apps import H1stAIModelingModuleConfig
from ..base import H1stModel


__all__: Sequence[str] = ('MLModel', 'H1stMLModel',

                          'PyLoadablePreTrainedMLModel',
                          'H1stPyLoadablePreTrainedMLModel')


class MLModel(H1stModel):
    artifact_global_url: CharField = \
        CharField(
            verbose_name='ML Model Artifact Global URL',
            help_text='ML Model Artifact Global URL',

            max_length=255,

            null=True,
            blank=True,
            choices=None,
            db_column=None,
            db_index=True,
            db_tablespace=None,
            default=None,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=True,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    artifact_local_path: CharField = \
        CharField(
            verbose_name='ML Model Artifact Local Path',
            help_text='ML Model Artifact Local Path',

            max_length=255,

            null=True,
            blank=True,
            choices=None,
            db_column=None,
            db_index=True,
            db_tablespace=None,
            default=None,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=False,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    native_model_obj: Optional[Any] = None

    class Meta(H1stModel.Meta):
        verbose_name: str = 'H1st ML Model'
        verbose_name_plural: str = 'H1st ML Models'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_ml_models'

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self.native_model_obj = None


class PyLoadablePreTrainedMLModel(MLModel):
    py_loader_module_and_qualname: CharField = \
        CharField(
            verbose_name='Pre-Trained ML Model Python Loader',
            help_text='Pre-Trained ML Model Python Loader (module.qualname)',

            max_length=255,

            null=False,
            blank=False,
            choices=None,
            db_column=None,
            db_index=True,
            db_tablespace=None,
            default=None,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=False,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    class Meta(MLModel.Meta):
        verbose_name: str = 'H1st Python-Loadable Pre-Trained ML Model'
        verbose_name_plural: str = 'H1st Python-Loadable Pre-Trained ML Models'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_py_loadable_pretrained_ml_models'

    @property
    def loader(self) -> callable:
        return import_obj(self.py_loader_module_and_qualname)

    @property
    def init_params(self) -> dict[str, Any]:
        return ({} if self.params is None else self.params).get('__init__', {})

    def load(self) -> None:
        if not self.native_model_obj:
            self.native_model_obj = self.loader(**self.init_params)


# alias
H1stMLModel = MLModel
H1stPyLoadablePreTrainedMLModel = PyLoadablePreTrainedMLModel

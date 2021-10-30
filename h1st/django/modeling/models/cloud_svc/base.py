from typing import Sequence   # TODO: Py3.9: use generic collections.abc

from ....util import PGSQL_IDENTIFIER_MAX_LEN
from ...apps import H1stAIModelingModuleConfig
from ..base import H1stModel


__all__: Sequence[str] = ('CloudServiceModel',)


class CloudServiceModel(H1stModel):
    class Meta(H1stModel.Meta):
        verbose_name: str = 'Cloud-Service Model'
        verbose_name_plural: str = 'Cloud-Service Models'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_cloud_svc_models'


# alias
H1stCloudServiceModel = CloudServiceModel

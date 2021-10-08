from ....util import PGSQL_IDENTIFIER_MAX_LEN
from ...apps import H1stAIModelingModuleConfig
from ..base import H1stModel


class CloudServiceModel(H1stModel):
    class Meta(H1stModel.Meta):
        verbose_name = 'Cloud-Service Model'
        verbose_name_plural = 'Cloud-Service Models'

        db_table = \
            f"{H1stAIModelingModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_cloud_svc_models'


# alias
H1stCloudServiceModel = CloudServiceModel

from typing import Sequence   # TODO: Py3.9: use generic collections.abc

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ..apps import H1stAIDataManagementModuleConfig
from .base import DataSet


__all__: Sequence[str] = ('LiveDataSource',)


class LiveDataSource(DataSet):
    class Meta(DataSet.Meta):
        verbose_name: str = 'Live Data Source'
        verbose_name_plural: str = 'Live Data Sources'

        db_table: str = (f'{H1stAIDataManagementModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'live_data_sources'

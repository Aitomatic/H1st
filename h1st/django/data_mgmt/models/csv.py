from typing import Sequence

import pandas

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ..apps import H1stAIDataManagementModuleConfig
from .base import DataSet


__all__: Sequence[str] = ('CSVDataSet',)


class CSVDataSet(DataSet):
    class Meta(DataSet.Meta):
        verbose_name: str = 'CSV Data Set'
        verbose_name_plural: str = 'CSV Data Sets'

        db_table: str = (f'{H1stAIDataManagementModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'csv_data_sets'

    def to_pandas(self, **kwargs) -> pandas.DataFrame:
        return pandas.read_csv(self.path, **kwargs)

    def load(self, **kwargs) -> None:
        self.native_data_obj = self.to_pandas(**kwargs)

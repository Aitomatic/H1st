from typing import Sequence   # TODO: Py3.9: use generic collections.abc

import pandas

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ..apps import H1stAIDataManagementModuleConfig
from .base import DataSet


__all__: Sequence[str] = ('ParquetDataSet',)


class ParquetDataSet(DataSet):
    class Meta(DataSet.Meta):
        verbose_name: str = 'Parquet Data Set'
        verbose_name_plural: str = 'Parquet Data Sets'

        db_table: str = (f'{H1stAIDataManagementModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'parquet_data_sets'

    def to_pandas(self, engine='pyarrow', columns=None, **kwargs):
        return pandas.read_parquet(path=self.path,
                                   engine=engine,
                                   columns=columns,
                                   **kwargs)

    def load(self, **kwargs) -> None:
        if self.native_data_obj is None:
            self.native_data_obj = self.to_pandas(**kwargs)

from typing import Sequence   # TODO: Py3.9: use generic collections.abc
import json

import pandas

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ..apps import H1stAIDataManagementModuleConfig
from .json import JSONDataSet


__all__: Sequence[str] = ('PandasDataFrame',)


class PandasDataFrame(JSONDataSet):
    class Meta(JSONDataSet.Meta):
        verbose_name: str = 'Pandas DataFrame'
        verbose_name_plural: str = 'Pandas DataFrames'

        db_table: str = (f'{H1stAIDataManagementModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'pandas_dataframes'

    @classmethod
    def jsonize(cls, df: pandas.DataFrame) -> dict:
        return json.loads(df.to_json(path_or_buf=None,
                                     orient='split',
                                     date_format='iso',
                                     double_precision=10,
                                     force_ascii=False,
                                     date_unit='ms',
                                     default_handler=None,
                                     lines=False,
                                     compression=None,
                                     index=True,
                                     indent=None,
                                     storage_options=None))

    def load(self) -> None:
        if self.native_data_obj is None:
            self.native_data_obj = pandas.DataFrame(**self.in_db_json)

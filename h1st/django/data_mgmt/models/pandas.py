import json

import pandas

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ..apps import H1stAIDataManagementModuleConfig
from .json import JSONDataSet


class PandasDataFrame(JSONDataSet):
    class Meta(JSONDataSet.Meta):
        verbose_name = 'Pandas DataFrame'
        verbose_name_plural = 'Pandas DataFrames'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'pandas_dataframes'

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

    def load(self):
        return pandas.DataFrame(**self.in_db_json)

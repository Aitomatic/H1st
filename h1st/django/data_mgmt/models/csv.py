import os

from django.conf import settings

import pandas

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ..apps import H1stAIDataManagementModuleConfig
from .base import DataSet


class CSVDataSet(DataSet):
    class Meta(DataSet.Meta):
        verbose_name = 'CSV Data Set'
        verbose_name_plural = 'CSV Data Sets'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'csv_data_sets'

    def to_pandas(self, **kwargs):
        # set AWS credentials if applicable
        aws_key = settings.__dict__.get('AWS_ACCESS_KEY_ID')
        aws_secret = settings.__dict__.get('AWS_SECRET_ACCESS_KEY')
        if aws_key and aws_secret:
            os.environ['AWS_ACCESS_KEY_ID'] = aws_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret

        return pandas.read_csv(self.path, **kwargs)

    def load(self):
        return self.to_pandas()

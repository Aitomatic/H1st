from collections.abc import Sequence

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ..apps import H1stAIDataManagementModuleConfig
from .base import DataSet


__all__: Sequence[str] = ('TFRecordDataSet',)


class TFRecordDataSet(DataSet):
    class Meta(DataSet.Meta):
        verbose_name: str = 'TFRecord Data Set'
        verbose_name_plural: str = 'TFRecord Data Sets'

        db_table: str = (f'{H1stAIDataManagementModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'tfrecord_data_sets'

from collections.abc import Sequence

from django.db.models.fields import TextField

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ..apps import H1stAIDataManagementModuleConfig
from .base import DataSet


__all__: Sequence[str] = ('TextDataSet',)


class TextDataSet(DataSet):
    in_db_text: TextField = \
        TextField(
            verbose_name='In-Database Text Data Content',
            help_text='In-Database Text Data Content',

            null=True,
            blank=True,
            choices=None,
            db_column=None,
            db_index=False,
            db_tablespace=None,
            default=None,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=False,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    class Meta(DataSet.Meta):
        verbose_name: str = 'Text Data Set'
        verbose_name_plural: str = 'Text Data Sets'

        db_table: str = (f'{H1stAIDataManagementModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'text_data_sets'

    def load(self):
        if self.native_data_obj is None:
            self.native_data_obj = self.in_db_text

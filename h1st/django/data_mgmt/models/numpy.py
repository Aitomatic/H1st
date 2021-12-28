from typing import Sequence   # TODO: Py3.9: use generic collections.abc
from json.decoder import JSONDecoder

from django.core.serializers.json import DjangoJSONEncoder
from django.db.models.fields.json import JSONField

import numpy

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ..apps import H1stAIDataManagementModuleConfig
from .json import JSONDataSet


__all__: Sequence[str] = ('NumPyArray',)


class NumPyArray(JSONDataSet):
    dtype: JSONField = \
        JSONField(
            verbose_name='Numpy Array Data Type(s)',
            help_text='Numpy Array Data Type(s)',

            encoder=DjangoJSONEncoder,
            decoder=JSONDecoder,

            null=False,
            blank=False,
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

    shape: JSONField = \
        JSONField(
            verbose_name='Numpy Array Shape',
            help_text='Numpy Array Shape',

            encoder=DjangoJSONEncoder,
            decoder=JSONDecoder,

            null=False,
            blank=False,
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

    class Meta(JSONDataSet.Meta):
        verbose_name: str = 'NumPy Array'
        verbose_name_plural: str = 'NumPy Arrays'

        db_table: str = (f'{H1stAIDataManagementModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'numpy_arrays'

    def load(self) -> None:
        if self.native_data_obj is None:
            self.native_data_obj = numpy.array(object=self.in_db_json,
                                               dtype=self.dtype,
                                               copy=False,
                                               order='K',
                                               subok=False,
                                               ndmin=0)

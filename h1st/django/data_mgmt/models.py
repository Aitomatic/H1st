import json
from json.decoder import JSONDecoder
import os

from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models.deletion import SET_NULL
from django.db.models.fields import BooleanField, CharField, TextField
from django.db.models.fields.json import JSONField
from django.db.models.fields.related import ForeignKey

from polymorphic.models import PolymorphicModel

import numpy
import pandas

from ..util import PGSQL_IDENTIFIER_MAX_LEN, dir_path_with_slash
from ..util.models import (
    _ModelWithUUIDPK,
    _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps
)
from .apps import H1stAIDataManagementModuleConfig


class DataSchema(PolymorphicModel, _ModelWithUUIDPK):
    name = \
        CharField(
            verbose_name='Data Schema Unique Name',
            help_text='Data Schema Unique Name',

            max_length=255,

            null=False,
            blank=False,
            choices=None,
            db_column=None,
            db_index=True,
            db_tablespace=None,
            default=None,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=True,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    specs = \
        JSONField(
            verbose_name='Data Schema Specifications',
            help_text='Data Schema Specifications',

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

    class Meta(_ModelWithUUIDPK.Meta):
        verbose_name = 'Data Schema'
        verbose_name_plural = 'Data Schemas'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'data_schemas'

        ordering = ('name',)

    def __str__(self) -> str:
        return f'"{self.name}" {type(self).__name__}'


class DataSet(PolymorphicModel,
              _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps):
    RELATED_NAME = 'data_sets'
    RELATED_QUERY_NAME = 'data_set'

    schema = \
        ForeignKey(
            verbose_name='Data Set Schema',
            help_text='Data Set Schema',

            # docs.djangoproject.com/en/dev/ref/models/fields/#arguments
            to=DataSchema,

            # docs.djangoproject.com/en/dev/ref/models/fields/#django.db.models.ForeignKey.on_delete
            on_delete=SET_NULL,

            # docs.djangoproject.com/en/dev/ref/models/fields/#django.db.models.ForeignKey.limit_choices_to
            limit_choices_to={},

            # docs.djangoproject.com/en/dev/ref/models/fields/#django.db.models.ForeignKey.related_name
            related_name=RELATED_NAME,

            # docs.djangoproject.com/en/dev/ref/models/fields/#django.db.models.ForeignKey.related_query_name
            related_query_name=RELATED_QUERY_NAME,

            # docs.djangoproject.com/en/dev/ref/models/fields/#django.db.models.ForeignKey.to_field
            # to_field=...,

            # docs.djangoproject.com/en/dev/ref/models/fields/#django.db.models.ForeignKey.db_constraint
            db_constraint=True,

            # docs.djangoproject.com/en/dev/ref/models/fields/#django.db.models.ForeignKey.swappable
            swappable=True,

            null=True,
            blank=True,
            choices=None,
            db_column=None,
            db_index=True,   # implied
            db_tablespace=None,
            default=None,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=False,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    class Meta(_ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps.Meta):
        verbose_name = 'Data Set'
        verbose_name_plural = 'Data Sets'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'data_sets'

    def __str__(self) -> str:
        return f'{type(self).__name__} #{self.uuid}'

    def load(self):
        raise NotImplementedError


class JSONDataSet(DataSet):
    json = \
        JSONField(
            verbose_name='JSON Data Content',
            help_text='JSON Data Content',

            encoder=DjangoJSONEncoder,
            decoder=JSONDecoder,

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
        verbose_name = 'JSON Data Set'
        verbose_name_plural = 'JSON Data Sets'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'json_data_sets'

    def load(self):
        return self.json


class NumPyArray(JSONDataSet):
    dtype = \
        JSONField(
            verbose_name='Data Type(s)',
            help_text='Data Type(s)',

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
        verbose_name = 'NumPy Array'
        verbose_name_plural = 'NumPy Arrays'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'numpy_arrays'

    def load(self):
        return numpy.array(object=self.json,
                           dtype=self.dtype,
                           copy=False,
                           order='K',
                           subok=False,
                           ndmin=0)


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
        return pandas.DataFrame(**self.json)


class TextDataSet(DataSet):
    text = \
        TextField(
            verbose_name='Text',
            help_text='Text',

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
        verbose_name = 'Text Data Set'
        verbose_name_plural = 'Text Data Sets'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'text_data_sets'

    def load(self):
        return self.text


class _FileStoredDataSet(DataSet):
    path = \
        CharField(
            verbose_name='Data Set Path/URL',
            help_text='Data Set Path/URL',

            max_length=255,

            null=False,
            blank=False,
            choices=None,
            db_column=None,
            db_index=True,
            db_tablespace=None,
            default=None,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=False,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    is_dir = \
        BooleanField(
            verbose_name='Data Set Path/URL is Directory?',
            help_text='Data Set Path/URL is Directory?',
            null=False,
            blank=False,
            choices=None,
            db_column=None,
            db_index=True,
            db_tablespace=None,
            default=False,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=False,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    class Meta(DataSet.Meta):
        abstract = True

    def __str__(self) -> str:
        return f'{type(self).__name__} #{self.uuid} @ ' \
            f'{dir_path_with_slash(self.path) if self.is_dir else self.path}'


class CSVDataSet(_FileStoredDataSet):
    class Meta(_FileStoredDataSet.Meta):
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


class ParquetDataSet(_FileStoredDataSet):
    class Meta(_FileStoredDataSet.Meta):
        verbose_name = 'Parquet Data Set'
        verbose_name_plural = 'Parquet Data Sets'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'parquet_data_sets'

    def to_pandas(self, engine='pyarrow', columns=None, **kwargs):
        # set AWS credentials if applicable
        aws_key = settings.__dict__.get('AWS_ACCESS_KEY_ID')
        aws_secret = settings.__dict__.get('AWS_SECRET_ACCESS_KEY')
        if aws_key and aws_secret:
            os.environ['AWS_ACCESS_KEY_ID'] = aws_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret

        return pandas.read_parquet(path=self.path,
                                   engine=engine,
                                   columns=columns,
                                   **kwargs)

    def load(self):
        return self.to_pandas()


class TFRecordDataSet(_FileStoredDataSet):
    class Meta(_FileStoredDataSet.Meta):
        verbose_name = 'TFRecord Data Set'
        verbose_name_plural = 'TFRecord Data Sets'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'tfrecord_data_sets'

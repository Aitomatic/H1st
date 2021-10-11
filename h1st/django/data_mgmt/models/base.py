__all__ = 'DataSchema', 'DataSet'


from abc import abstractmethod
from json.decoder import JSONDecoder

from django.core.serializers.json import DjangoJSONEncoder
from django.db.models.deletion import SET_NULL
from django.db.models.fields import BooleanField, CharField
from django.db.models.fields.json import JSONField
from django.db.models.fields.related import ForeignKey

from polymorphic.models import PolymorphicModel

from ...util import PGSQL_IDENTIFIER_MAX_LEN, dir_path_with_slash
from ...util.models import (_ModelWithUUIDPK,
                            _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps)
from ..apps import H1stAIDataManagementModuleConfig


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

    global_url = \
        CharField(
            verbose_name='Data Set Global URL',
            help_text='Data Set Global URL',

            max_length=255,

            null=True,
            blank=True,
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

    local_path = \
        CharField(
            verbose_name='Data Set Local Path',
            help_text='Data Set Local Path',

            max_length=255,

            null=True,
            blank=True,
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

    path_is_dir = \
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

    native_data_obj = None

    class Meta(_ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps.Meta):
        verbose_name = 'Data Set'
        verbose_name_plural = 'Data Sets'

        db_table = (f'{H1stAIDataManagementModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'data_sets'

    @property
    def _path_repr(self) -> str:
        global_url_repr = ((dir_path_with_slash(self.global_url)
                            if self.path_is_dir
                            else self.global_url)
                           if self.global_url
                           else None)

        local_path_repr = ((dir_path_with_slash(self.local_path)
                            if self.path_is_dir
                            else self.local_path)
                           if self.local_path
                           else None)

        return ((f' @ {global_url_repr} (local: {local_path_repr})'
                 if local_path_repr
                 else f' @ {global_url_repr}')
                if global_url_repr
                else (f' @ {local_path_repr}'
                      if local_path_repr
                      else ''))

    def __str__(self) -> str:
        return f'{type(self).__name__} #{self.uuid}{self._path_repr}'

    @abstractmethod
    def load(self):
        raise NotImplementedError

    def unload(self):
        self.native_data_obj = None

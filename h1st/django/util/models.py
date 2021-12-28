"""H1st Django DB Model Utilities."""


from __future__ import annotations

from typing import List, Sequence   # TODO: Py3.9: use generics/collections.abc
from typing import Union
from uuid import UUID, uuid4

from django.db.models.base import Model
from django.db.models.constraints import BaseConstraint
from django.db.models.fields import CharField, UUIDField
from django.db.models.indexes import Index
from django.utils.functional import classproperty

from model_utils.models import TimeStampedModel


__all__: Sequence[str] = ('_ModelWithUUIDPK',
                          '_ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps')


class _ModelWithUUIDPK(Model):
    uuid: UUIDField = \
        UUIDField(
            # docs.djangoproject.com/en/dev/ref/models/fields/#field-options

            # docs.djangoproject.com/en/dev/ref/models/fields/#verbose-name
            verbose_name='UUID',

            # docs.djangoproject.com/en/dev/ref/models/fields/#help-text
            help_text='UUID',

            # docs.djangoproject.com/en/dev/ref/models/fields/#null
            null=False,   # implied

            # docs.djangoproject.com/en/dev/ref/models/fields/#blank
            blank=False,

            # docs.djangoproject.com/en/dev/ref/models/fields/#choices
            choices=None,

            # docs.djangoproject.com/en/dev/ref/models/fields/#db-column
            db_column=None,

            # docs.djangoproject.com/en/dev/ref/models/fields/#db-index
            db_index=True,

            # docs.djangoproject.com/en/dev/ref/models/fields/#db-tablespace
            db_tablespace=None,

            # docs.djangoproject.com/en/dev/ref/models/fields/#default
            default=uuid4,

            # docs.djangoproject.com/en/dev/ref/models/fields/#editable
            editable=False,

            # docs.djangoproject.com/en/dev/ref/models/fields/#error-messages
            # error_messages=None,

            # docs.djangoproject.com/en/dev/ref/models/fields/#primary-key
            primary_key=True,

            # docs.djangoproject.com/en/dev/ref/models/fields/#unique
            unique=True,   # implied

            # docs.djangoproject.com/en/dev/ref/models/fields/#unique-for-date
            unique_for_date=None,

            # docs.djangoproject.com/en/dev/ref/models/fields/#unique-for-month
            unique_for_month=None,

            # docs.djangoproject.com/en/dev/ref/models/fields/#unique-for-year
            unique_for_year=None,

            # docs.djangoproject.com/en/dev/ref/models/fields/#validators
            # validators=None
        )

    # docs.djangoproject.com/en/dev/ref/models/options/#available-meta-options
    class Meta:
        # docs.djangoproject.com/en/dev/ref/models/options/#abstract
        abstract: bool = True

        # docs.djangoproject.com/en/dev/ref/models/options/#app-label
        # app_label: str = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#base-manager-name
        base_manager_name: str = 'objects'

        # docs.djangoproject.com/en/dev/ref/models/options/#db-table
        # db_table: str = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#db-tablespace
        # db_tablespace: str = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#default-manager-name
        default_manager_name: str = 'objects'

        # docs.djangoproject.com/en/dev/ref/models/options/#default-related-name
        # default_related_name: str = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#get-latest-by
        # get_latest_by: Union[str, Sequence[str]] = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#managed
        managed: bool = True

        # docs.djangoproject.com/en/dev/ref/models/options/#order-with-respect-to
        # order_with_respect_to: str = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#ordering
        ordering: Sequence[str] = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#permissions
        permissions: Sequence[tuple[str, str]] = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#default-permissions
        default_permissions: Sequence[str] = 'add', 'change', 'delete', 'view'

        # docs.djangoproject.com/en/dev/ref/models/options/#proxy
        proxy: bool = False

        # docs.djangoproject.com/en/dev/ref/models/options/#required-db-features
        required_db_features: Sequence[str] = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#required-db-vendor
        # required_db_vendor: str = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#select-on-save
        select_on_save: bool = False

        # docs.djangoproject.com/en/dev/ref/models/options/#indexes
        indexes: Sequence[Index] = ()

        # DEPRECATED
        # docs.djangoproject.com/en/dev/ref/models/options/#unique-together
        # unique_together: Sequence[Union[str, Sequence[str]]] = ()

        # DEPRECATED
        # docs.djangoproject.com/en/dev/ref/models/options/#index-together
        # index_together: Sequence[Union[str, Sequence[str]]] = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#constraints
        constraints: Sequence[BaseConstraint] = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#verbose-name
        # verbose_name: str = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#verbose-name-plural
        # verbose_name_plural: str  = ...


class _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps(_ModelWithUUIDPK,
                                                         TimeStampedModel):
    name: CharField = \
        CharField(
            verbose_name='(optional) Unique Name',
            help_text='(optional) Unique Name',

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
            unique=True,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    class Meta(_ModelWithUUIDPK.Meta):
        abstract: bool = True

        get_latest_by: Union[str, Sequence[str]] = 'modified'

        ordering: Sequence[str] = 'name', '-modified'

    def __str__(self) -> str:
        return (f'{type(self).__name__} "{self.name}"'
                if self.name
                else f'{type(self).__name__} #{self.uuid}')

    @property
    def name_or_uuid(self) -> str:
        return self.name if self.name else self.uuid

    @classproperty
    def names_or_uuids(cls) -> List[str]:
        return [(name if name else uuid)
                for name, uuid in cls.objects.values_list('name', 'uuid')]

    @classmethod
    def get_by_name_or_uuid(cls, name_or_uuid: Union[str, UUID]) \
            -> _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps:
        try:   # try looking up object by UUID
            _uuid: UUID = UUID(hex=name_or_uuid, version=4)
            return cls.objects.get(uuid=_uuid)

        except ValueError:
            # else look up by Name
            return cls.objects.get(name=name_or_uuid)

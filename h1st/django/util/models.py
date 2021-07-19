from uuid import uuid4

from django.db.models.base import Model
from django.db.models.fields import CharField, UUIDField

from model_utils.models import TimeStampedModel


class _ModelWithUUIDPK(Model):
    uuid = \
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
        abstract = True

        # docs.djangoproject.com/en/dev/ref/models/options/#app-label
        # app_label = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#base-manager-name
        base_manager_name = 'objects'

        # docs.djangoproject.com/en/dev/ref/models/options/#db-table
        # db_table = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#db-tablespace
        # db_tablespace = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#default-manager-name
        default_manager_name = 'objects'

        # docs.djangoproject.com/en/dev/ref/models/options/#default-related-name
        # default_related_name = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#get-latest-by
        # get_latest_by = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#managed
        managed = True

        # docs.djangoproject.com/en/dev/ref/models/options/#order-with-respect-to
        # order_with_respect_to = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#ordering
        ordering = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#permissions
        permissions = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#default-permissions
        default_permissions = 'add', 'change', 'delete', 'view'

        # docs.djangoproject.com/en/dev/ref/models/options/#proxy
        proxy = False

        # docs.djangoproject.com/en/dev/ref/models/options/#required-db-features
        required_db_features = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#required-db-vendor
        # required_db_vendor = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#select-on-save
        select_on_save = False

        # docs.djangoproject.com/en/dev/ref/models/options/#indexes
        indexes = ()

        # DEPRECATED
        # docs.djangoproject.com/en/dev/ref/models/options/#unique-together
        # unique_together = ()

        # DEPRECATED
        # docs.djangoproject.com/en/dev/ref/models/options/#index-together
        # index_together = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#constraints
        constraints = ()

        # docs.djangoproject.com/en/dev/ref/models/options/#verbose-name
        # verbose_name = ...

        # docs.djangoproject.com/en/dev/ref/models/options/#verbose-name-plural
        # verbose_name_plural = ...


class _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps(
        _ModelWithUUIDPK,
        TimeStampedModel):
    name = \
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
        abstract = True

        get_latest_by = 'modified'

        ordering = 'name', '-modified',
    
    def __str__(self) -> str:
        return (f'{type(self).__name__} "{self.name}"'
                if self.name
                else f'{type(self).__name__} #{self.uuid}')

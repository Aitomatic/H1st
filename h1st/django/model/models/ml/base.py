from django.db.models.fields import CharField

from ....util import PGSQL_IDENTIFIER_MAX_LEN
from ...apps import H1stModelModuleConfig
from ..base import H1stModel


class MLModel(H1stModel):
    artifact_local_path = \
        CharField(
            verbose_name='ML Model Artifact Local Path',
            help_text='ML Model Artifact Local Path',

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

    class Meta(H1stModel.Meta):
        verbose_name = 'H1st ML Model'
        verbose_name_plural = 'H1st ML Models'

        db_table = \
            f"{H1stModelModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_ml_models'


# alias
H1stMLModel = MLModel

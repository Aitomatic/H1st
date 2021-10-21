from json.decoder import JSONDecoder

from django.core.serializers.json import DjangoJSONEncoder
from django.db.models.fields.json import JSONField

from polymorphic.models import PolymorphicModel

from django_plotly_dash import DjangoDash
from gradio.interface import Interface

from h1st.model.model import Model as CoreH1stModel

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ...util.models import _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps
from ..apps import H1stAIModelingModuleConfig


class Model(PolymorphicModel,
            _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps,
            CoreH1stModel):
    params = \
        JSONField(
            verbose_name='Model Parameters',
            help_text='Model Parameters',

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

    class Meta(_ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps.Meta):
        verbose_name = 'H1st Model'
        verbose_name_plural = 'H1st Models'

        db_table = (f'{H1stAIModelingModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_models'

    @property
    def dash_ui(self) -> DjangoDash:
        return NotImplemented

    @property
    def gradio_ui(self) -> Interface:
        return NotImplemented


# aliases
H1stModel = Model

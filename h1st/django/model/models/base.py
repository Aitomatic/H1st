from polymorphic.models import PolymorphicModel

from h1st.model.model import Model as CoreH1stModel

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ...util.models import _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps
from ..apps import H1stModelModuleConfig


class Model(PolymorphicModel,
            _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps,
            CoreH1stModel):
    class Meta(_ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps.Meta):
        verbose_name = 'H1st Model'
        verbose_name_plural = 'H1st Models'

        db_table = \
            f"{H1stModelModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_models'


# aliases
H1stModel = Model

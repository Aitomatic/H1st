from polymorphic.models import PolymorphicModel

from h1st.model.model import Model as _CoreH1stModel

from ...util import PGSQL_IDENTIFIER_MAX_LEN
from ...util.models import _ModelWithUUIDPKAndTimestamps, _ModelWithUniqueName
from ..apps import H1stModelModuleConfig


class Model(PolymorphicModel, _ModelWithUUIDPKAndTimestamps, _CoreH1stModel):
    class Meta(_ModelWithUUIDPKAndTimestamps.Meta):
        verbose_name = 'H1st Model'
        verbose_name_plural = 'H1st Models'

        db_table = \
            f"{H1stModelModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_models'

    def __str__(self) -> str:
        return f'{type(self).__name__} #{self.uuid}'


class NamedModel(_ModelWithUniqueName, Model):
    class Meta(_ModelWithUniqueName.Meta, Model.Meta):
        verbose_name = 'Named H1st Model'
        verbose_name_plural = 'Named H1st Models'

        db_table = \
            f"{H1stModelModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'named_h1st_models'

    def __str__(self) -> str:
        return f'{type(self).__name__} "{self.name}"'


# aliases
H1stModel = Model
NamedH1stModel = NamedModel

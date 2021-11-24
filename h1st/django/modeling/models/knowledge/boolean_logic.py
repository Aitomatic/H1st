"""Boolean Logic Knowledge Model."""


import sys

from h1st.django.util import PGSQL_IDENTIFIER_MAX_LEN
from ...apps import H1stAIModelingModuleConfig
from ..base import H1stModel

from .base import KnowledgeModel

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = (
    'BooleanLogicKnowledgeModel',
)


class BooleanLogicKnowledgeModel(KnowledgeModel):
    """Boolean Logic Knowledge Model."""

    class Meta(H1stModel.Meta):
        """Metadata."""

        verbose_name: str = 'H1st Boolean Logic Knowledge Model'
        verbose_name_plural: str = 'H1st Boolean Logic Knowledge Models'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_boolean_logic_knowledge_models'

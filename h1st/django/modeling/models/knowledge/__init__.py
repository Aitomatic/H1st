"""Knowledge Models."""


import sys

from .base import KnowledgeModel
from .boolean_logic import BooleanLogicKnowledgeModel

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = (
    'KnowledgeModel',
    'BooleanLogicKnowledgeModel',
)

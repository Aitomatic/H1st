"""Knowledge Modelers."""


import sys

from .boolean_logic import BooleanLogicKnowledgeModeler
from .fuzzy_logic import FuzzyLogicKnowledgeModeler

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = (
    'BooleanLogicKnowledgeModeler',
    'FuzzyLogicKnowledgeModeler',
)

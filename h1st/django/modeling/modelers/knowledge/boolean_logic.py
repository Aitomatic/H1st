"""Boolean Logic Knowledge Modeler."""


import sys

from ..base import H1stModeler

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = (
    'BooleanLogicKnowledgeModeler',
)


class BooleanLogicKnowledgeModeler(H1stModeler):
    """Boolean Logic Knowledge Modeler."""

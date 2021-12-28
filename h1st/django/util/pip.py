"""H1st PIP-related Utilities."""


from typing import Dict, List, Sequence   # TODO: Py3.9: use generics
from typing import Optional

from pip._internal.operations.freeze import freeze


__all__: Sequence[str] = ('get_python_dependencies',)


def get_python_dependencies() -> Dict[str, Optional[str]]:
    """Get Python dependencies and their versions as a dictionary."""

    d: Dict[str, Optional[str]] = {}

    for deps_and_vers in freeze():
        ls: List[str] = deps_and_vers.split('==')

        if len(ls) == 2:
            d[ls[0]] = ls[1]

        else:
            assert len(ls) == 1, f'*** {ls} ***'
            d[ls[0]] = None

    return d

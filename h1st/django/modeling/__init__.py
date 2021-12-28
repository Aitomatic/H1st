"""H1st Django Modeling module."""


import sys

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = ('default_app_config',)


# pylint: disable=invalid-name
default_app_config = 'h1st.django.modeling.apps.H1stAIModelingModuleConfig'

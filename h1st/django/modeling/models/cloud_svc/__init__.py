from typing import Sequence   # TODO: Py3.9: use generic collections.abc

from .base import CloudServiceModel, H1stCloudServiceModel

from .google.translation import (GoogleCloudTranslationServiceModel,
                                 H1stGoogleCloudTranslationServiceModel,

                                 GoogleTranslateServiceModel,
                                 H1stGoogleTranslateServiceModel)


__all__: Sequence[str] = (
    'CloudServiceModel', 'H1stCloudServiceModel',

    'GoogleCloudTranslationServiceModel',
    'H1stGoogleCloudTranslationServiceModel',

    'GoogleTranslateServiceModel', 'H1stGoogleTranslateServiceModel',
)

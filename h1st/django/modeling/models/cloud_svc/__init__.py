from collections.abc import Sequence

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

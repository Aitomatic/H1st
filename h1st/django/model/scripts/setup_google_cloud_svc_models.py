from ..models import (GoogleCloudTranslationServiceModel,
                      GoogleTranslateServiceModel)


def run():
    print(GoogleCloudTranslationServiceModel.objects.get_or_create(
            name='Google-Cloud-Translation-Service-Model')[0])

    print(GoogleTranslateServiceModel.objects.get_or_create(
            name='Google-Translate-Service-Model')[0])

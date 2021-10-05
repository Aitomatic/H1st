from google.cloud import translate_v2 as translate

from .....util import PGSQL_IDENTIFIER_MAX_LEN
from ....apps import H1stModelModuleConfig
from .. import CloudServiceModel


class GoogleCloudTranslationServiceModel(CloudServiceModel):
    class Meta(CloudServiceModel.Meta):
        verbose_name = 'Google Cloud Translation Service Model'
        verbose_name_plural = 'Google Cloud Translation Service Models'

        db_table = \
            f"{H1stModelModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_google_cloud_translation_service_models'

    def predict(self, text, src_lang='en', target_lang='es'):
        translate_client = translate.Client()

        result = translate_client.translate(text,
                                            source_language=src_lang,
                                            target_language=target_lang)

        return result['translatedText']


class GoogleTranslateServiceModel(CloudServiceModel):
    class Meta(CloudServiceModel.Meta):
        verbose_name = 'Google Translate Service Model'
        verbose_name_plural = 'Google Translate Service Models'

        db_table = \
            f"{H1stModelModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_google_translate_service_models'


# aliases
H1stGoogleCloudTranslationServiceModel = GoogleCloudTranslationServiceModel
H1stGoogleTranslateServiceModel = GoogleTranslateServiceModel

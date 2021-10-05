from django.contrib.admin.decorators import register
from django.contrib.admin.options import ModelAdmin
from django.contrib.admin.sites import site

from silk.profiling.profiler import silk_profile

from .models import (
    GoogleCloudTranslationServiceModel, GoogleTranslateServiceModel,
    PreTrainedKerasImageNetClassifier,
)


@register(GoogleCloudTranslationServiceModel, site=site)
class GoogleCloudTranslationServiceModelAdmin(ModelAdmin):
    show_full_result_count = False

    @silk_profile(
        name=(f'{__module__}: '
              f'{GoogleCloudTranslationServiceModel._meta.verbose_name}'))
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=(f'{__module__}: '
              f'{GoogleCloudTranslationServiceModel._meta.verbose_name_plural}'
              ))
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(GoogleTranslateServiceModel, site=site)
class GoogleTranslateServiceModelAdmin(ModelAdmin):
    show_full_result_count = False

    @silk_profile(
        name=(f'{__module__}: '
              f'{GoogleTranslateServiceModel._meta.verbose_name}'))
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=(f'{__module__}: '
              f'{GoogleTranslateServiceModel._meta.verbose_name_plural}'))
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(PreTrainedKerasImageNetClassifier, site=site)
class PreTrainedKerasImageNetClassifierAdmin(ModelAdmin):
    show_full_result_count = False

    @silk_profile(
        name=(f'{__module__}: '
              f'{PreTrainedKerasImageNetClassifier._meta.verbose_name}'))
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=(f'{__module__}: '
              f'{PreTrainedKerasImageNetClassifier._meta.verbose_name_plural}')
    )
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)

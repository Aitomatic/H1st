from django.contrib.admin.decorators import register
from django.contrib.admin.options import ModelAdmin
from django.contrib.admin.sites import site

from silk.profiling.profiler import silk_profile

from .models import (
    GoogleCloudTranslationServiceModel, GoogleTranslateServiceModel,
    PreTrainedKerasImageNetClassifier, PreTrainedTorchVisionImageNetClassifier,
    PreTrainedHuggingFaceTransformer,
)


class H1stModelAdmin(ModelAdmin):
    readonly_fields = ('gradio_interactive_ui',)

    def gradio_interactive_ui(self, obj):
        return ('(not implemented)'
                if obj.gradio_ui is NotImplemented
                else f'h1st/models/{obj.name_or_uuid}/ui')


@register(GoogleCloudTranslationServiceModel, site=site)
class GoogleCloudTranslationServiceModelAdmin(H1stModelAdmin):
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
class GoogleTranslateServiceModelAdmin(H1stModelAdmin):
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
class PreTrainedKerasImageNetClassifierAdmin(H1stModelAdmin):
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


@register(PreTrainedTorchVisionImageNetClassifier, site=site)
class PreTrainedTorchVisionImageNetClassifierAdmin(H1stModelAdmin):
    show_full_result_count = False

    @silk_profile(
        name=(f'{__module__}: '
              f'{PreTrainedTorchVisionImageNetClassifier._meta.verbose_name}'))
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        f'{__module__}: '
        f'{PreTrainedTorchVisionImageNetClassifier._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(PreTrainedHuggingFaceTransformer, site=site)
class PreTrainedHuggingFaceTransformerAdmin(H1stModelAdmin):
    show_full_result_count = False

    @silk_profile(
        name=(f'{__module__}: '
              f'{PreTrainedHuggingFaceTransformer._meta.verbose_name}'))
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=(f'{__module__}: '
              f'{PreTrainedHuggingFaceTransformer._meta.verbose_name_plural}'))
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)

from typing import Sequence   # TODO: Py3.9: use generic collections.abc

from django.contrib.admin.decorators import register
from django.contrib.admin.options import ModelAdmin
from django.contrib.admin.sites import site

from silk.profiling.profiler import silk_profile

from .models import Decision, ModelEvalMetricsSet


__all__: Sequence[str] = 'DecisionAdmin', 'ModelEvalMetricsSetAdmin'


@register(Decision, site=site)
class DecisionAdmin(ModelAdmin):
    show_full_result_count = False

    @silk_profile(name=f'{__module__}: {Decision._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(name=f'{__module__}: {Decision._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(ModelEvalMetricsSet, site=site)
class ModelEvalMetricsSetAdmin(ModelAdmin):
    show_full_result_count = False

    @silk_profile(
        name=f'{__module__}: {ModelEvalMetricsSet._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=f'{__module__}: {ModelEvalMetricsSet._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)

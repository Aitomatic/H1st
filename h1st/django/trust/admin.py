from django.contrib.admin.decorators import register
from django.contrib.admin.options import ModelAdmin
from django.contrib.admin.sites import site

from .models import Decision, ModelEvalMetricsSet


@register(Decision, site=site)
class DecisionAdmin(ModelAdmin):
    show_full_result_count = False

    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(ModelEvalMetricsSet, site=site)
class ModelEvalMetricsSetAdmin(ModelAdmin):
    show_full_result_count = False

    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)

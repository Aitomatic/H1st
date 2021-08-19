from django.contrib.admin.decorators import register
from django.contrib.admin.options import ModelAdmin
from django.contrib.admin.sites import site

from silk.profiling.profiler import silk_profile

from .models import (
    DataSchema,

    JSONDataSet,
    NumPyArray,
    PandasDataFrame,

    TextDataSet,

    CSVDataSet,
    ParquetDataSet,
    TFRecordDataSet
)


@register(DataSchema, site=site)
class DataSchemaAdmin(ModelAdmin):
    show_full_result_count = False

    @silk_profile(name=f'{__module__}: {DataSchema._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(name=f'{__module__}: {DataSchema._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(JSONDataSet, site=site)
class JSONDataSetAdmin(ModelAdmin):
    show_full_result_count = False

    def get_queryset(self, request):
        query_set = super().get_queryset(request=request)

        return (query_set
                if request.resolver_match.url_name.endswith('_change')
                else query_set.defer('json')   # defer often large JSON field
                )

    @silk_profile(name=f'{__module__}: {JSONDataSet._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=f'{__module__}: {JSONDataSet._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(NumPyArray, site=site)
class NumPyArrayAdmin(JSONDataSetAdmin):
    @silk_profile(name=f'{__module__}: {NumPyArray._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=f'{__module__}: {NumPyArray._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(PandasDataFrame, site=site)
class PandasDataFrameAdmin(JSONDataSetAdmin):
    @silk_profile(name=f'{__module__}: {PandasDataFrame._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=f'{__module__}: {PandasDataFrame._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(TextDataSet, site=site)
class TextDataSetAdmin(ModelAdmin):
    show_full_result_count = False

    def get_queryset(self, request):
        query_set = super().get_queryset(request=request)

        return (query_set
                if request.resolver_match.url_name.endswith('_change')
                else query_set.defer('text')   # defer often large text field
                )

    @silk_profile(name=f'{__module__}: {TextDataSet._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=f'{__module__}: {TextDataSet._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(CSVDataSet, site=site)
class CSVDataSetAdmin(ModelAdmin):
    show_full_result_count = False

    @silk_profile(
        name=f'{__module__}: {CSVDataSet._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=f'{__module__}: {CSVDataSet._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(ParquetDataSet, site=site)
class ParquetDataSetAdmin(ModelAdmin):
    show_full_result_count = False

    @silk_profile(
        name=f'{__module__}: {ParquetDataSet._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=f'{__module__}: {ParquetDataSet._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)


@register(TFRecordDataSet, site=site)
class TFRecordDataSetAdmin(ModelAdmin):
    show_full_result_count = False

    @silk_profile(
        name=f'{__module__}: {TFRecordDataSet._meta.verbose_name}')
    def changeform_view(self, *args, **kwargs):
        return super().changeform_view(*args, **kwargs)

    @silk_profile(
        name=f'{__module__}: {TFRecordDataSet._meta.verbose_name_plural}')
    def changelist_view(self, *args, **kwargs):
        return super().changelist_view(*args, **kwargs)

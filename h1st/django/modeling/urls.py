from typing import Sequence   # TODO: Py3.9: use generic collections.abc

from django.urls.conf import include, path
from django.urls.resolvers import URLPattern

from rest_framework.routers import BaseRouter, DefaultRouter

from .api.rest.views import H1stModelViewSet, ModelExecAPIView
from .views import launch_gradio_ui


__all__: Sequence[str] = ('urlpatterns',)


CORE_REST_ROUTER: BaseRouter = DefaultRouter(trailing_slash=False)

CORE_REST_ROUTER.register(prefix='', viewset=H1stModelViewSet, basename=None)


urlpatterns: Sequence[URLPattern] = (
    path(route='',
         view=include(CORE_REST_ROUTER.urls),
         kwargs=None, name=None),

    path(route='<str:model_name_or_uuid>/exec/',
         view=ModelExecAPIView.as_view(),
         kwargs=None, name=None),

    path(route='<str:model_class_or_instance_name_or_uuid>/gradio/',
         view=launch_gradio_ui,
         kwargs=None, name=None),
)

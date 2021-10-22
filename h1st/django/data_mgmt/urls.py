from collections.abc import Sequence

from django.urls.conf import include, path
from django.urls.resolvers import URLPattern

from rest_framework.routers import DefaultRouter

from .api.rest.views import DataSetViewSet, DataQueryAPIView


__all__: Sequence[str] = ('urlpatterns',)


CORE_REST_API_ROUTER = DefaultRouter(trailing_slash=False)

CORE_REST_API_ROUTER.register(prefix='', viewset=DataSetViewSet, basename=None)


urlpatterns: Sequence[URLPattern] = (
    path(route='',
         view=include(CORE_REST_API_ROUTER.urls),
         kwargs=None, name=None),

    path(route='query/',
         view=DataQueryAPIView.as_view(),
         kwargs=None, name=None),
)

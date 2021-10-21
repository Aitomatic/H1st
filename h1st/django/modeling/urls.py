from django.urls.conf import include, path

from rest_framework.routers import DefaultRouter

from .api.rest.views import H1stModelViewSet, ModelExecAPIView
from .views import exec_on_json_input_data, launch_gradio_ui


CORE_REST_API_ROUTER = DefaultRouter(trailing_slash=False)

CORE_REST_API_ROUTER.register(
    prefix='',
    viewset=H1stModelViewSet,
    basename=None)


urlpatterns = [
    path(route='',
         view=include(CORE_REST_API_ROUTER.urls)),

    path(route='<str:model_name_or_uuid>/exec/',
         view=ModelExecAPIView.as_view()),

    path(route='<str:model_class_or_instance_name_or_uuid>/gradio/',
         view=launch_gradio_ui),

    path(route='<str:model_name_or_uuid>/<str:json_input_data>/',
         view=exec_on_json_input_data),
]

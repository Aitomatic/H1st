from inspect import getsource
from tempfile import mkstemp

from django.core.files.uploadedfile import (
    InMemoryUploadedFile,
    TemporaryUploadedFile
)

from rest_framework.authentication import (
    BasicAuthentication,
    RemoteUserAuthentication,
    SessionAuthentication,
    TokenAuthentication
)
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.renderers import CoreJSONRenderer, JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from rest_framework.filters import OrderingFilter
from rest_framework_filters.backends import \
    ComplexFilterBackend, \
    RestFrameworkFilterBackend

from silk.profiling.profiler import silk_profile

from ....data.util import \
    load_data_set_pointers_as_json, \
    save_numpy_arrays_and_pandas_dfs_as_data_set_pointers
from ...models import Model
from .filters import ModelFilter
from .queries import MODEL_REST_API_QUERY_SET
from .serializers import H1stModelSerializer
from ....trust.models import Decision
from ....util.views import LookUpByUUIDorNameMixin


class H1stModelViewSet(LookUpByUUIDorNameMixin, ModelViewSet):
    queryset = MODEL_REST_API_QUERY_SET

    serializer_class = H1stModelSerializer

    authentication_classes = (
        BasicAuthentication,
        RemoteUserAuthentication,
        SessionAuthentication,
        TokenAuthentication
    )

    permission_classes = (IsAuthenticated,)

    filter_class = ModelFilter

    filter_backends = \
        OrderingFilter, \
        ComplexFilterBackend, \
        RestFrameworkFilterBackend

    ordering_fields = 'name', 'created', 'modified'

    ordering = 'name', '-modified'

    pagination_class = None

    parser_classes = (JSONParser,)

    renderer_classes = \
        CoreJSONRenderer, \
        JSONRenderer

    lookup_fields = 'uuid', 'name'

    @silk_profile(name=f'{__module__}: {Model._meta.verbose_name_plural}')
    def list(self, *args, **kwargs):
        return super().list(*args, **kwargs)

    @silk_profile(name=f'{__module__}: {Model._meta.verbose_name}')
    def retrieve(self, *args, **kwargs):
        return super().retrieve(*args, **kwargs)


class ModelExecAPIView(APIView):
    authentication_classes = (BasicAuthentication,
                              RemoteUserAuthentication,
                              SessionAuthentication,
                              TokenAuthentication)

    permission_classes = (IsAuthenticated,)

    parser_classes = JSONParser, MultiPartParser

    @staticmethod
    def handle_request_data_value(v):
        if isinstance(v, InMemoryUploadedFile):
            tmp_file_handle, tmp_file_path = mkstemp()
            tmp_file_handle.write(v.read())
            return tmp_file_path

        elif isinstance(v, TemporaryUploadedFile):
            return v.temporary_file_path()

        else:
            return v

    def post(self, request, model_name_or_uuid: str):
        model = Model.get_by_name_or_uuid(name_or_uuid=model_name_or_uuid)

        if request.content_type == 'application/json':
            json_input_data = request.data

            loaded_json_input_data = \
                load_data_set_pointers_as_json(json_input_data)

            json_output_data = model.predict(loaded_json_input_data)

            saved_json_output_data = \
                save_numpy_arrays_and_pandas_dfs_as_data_set_pointers(
                    json_output_data)

            Decision.objects.create(
                input_data=json_input_data,
                model=model,
                model_code={str(model.uuid): getsource(type(model))},
                output_data=saved_json_output_data)

            return Response(data=saved_json_output_data,
                            status=None,
                            template_name=None,
                            headers=None,
                            exception=False,
                            content_type=None)

        elif request.content_type.startswith('multipart/form-data'):
            data = {}

            print(f'*** REQUEST DATA: {request.data} ***')

            for k, v in request.data.items():
                v_list = request.data.getlist(key=k)
                data[k] = ([self.handle_request_data_value(v) for v in v_list]
                           if len(v_list) > 1
                           else self.handle_request_data_value(v))

            print(f'*** MODEL INPUT DATA: {data} ***')

            json_output_data = model.predict(**data)

            saved_json_output_data = \
                save_numpy_arrays_and_pandas_dfs_as_data_set_pointers(
                    json_output_data)

            print(f'*** MODEL OUTPUT: {saved_json_output_data} ***')

            Decision.objects.create(
                input_data=data,
                model=model,
                model_code={str(model.uuid): getsource(type(model))},
                output_data=saved_json_output_data)

            return Response(data=saved_json_output_data,
                            status=None,
                            template_name=None,
                            headers=None,
                            exception=False,
                            content_type=None)

        else:
            return Response('Content Type must be '
                            "either 'application/json' "
                            "or 'multipart/form-data'")

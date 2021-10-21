from inspect import getsource
import json
from typing import Union

from django.core.serializers.json import DjangoJSONEncoder
from django.http.response import (HttpResponse, Http404,
                                  HttpResponseRedirect, JsonResponse)

from gradio.inputs import Dropdown
from gradio.interface import Interface

from ..data_mgmt.util import (
    load_data_set_pointers_as_json,
    save_numpy_arrays_and_pandas_dfs_as_data_set_pointers)
from .models import Model
from ..trust_vault.models import Decision


def launch_gradio_ui(request, model_class_or_instance_name_or_uuid: str) \
        -> Union[HttpResponse, Http404]:
    model_subclasses_by_name = Model.subclasses_by_name

    if model_class_or_instance_name_or_uuid in model_subclasses_by_name:
        model = model_subclasses_by_name[model_class_or_instance_name_or_uuid]

        model_names_or_uuids = model.names_or_uuids

        if not model_names_or_uuids:
            return Http404('*** MODEL CLASS ' +
                           model_class_or_instance_name_or_uuid +
                           ' HAS NO INSTANCES ***')

    else:
        try:
            model = Model.get_by_name_or_uuid(
                name_or_uuid=model_class_or_instance_name_or_uuid)

        except Model.DoesNotExist:
            return Http404('*** MODEL INSTANCE ' +
                           model_class_or_instance_name_or_uuid +
                           ' NOT FOUND ***')

        model_names_or_uuids = model.names_or_uuids

    gradio_interface = model.gradio_ui
    assert isinstance(gradio_interface, Interface), \
        TypeError(f'*** {gradio_interface} NOT A GRADIO INTERFACE ***')

    assert isinstance(gradio_interface.predict, list), \
        TypeError(f'*** {gradio_interface.predict} NOT A LIST ***')
    f = gradio_interface.predict.pop()
    assert not gradio_interface.predict, \
        ValueError(f'*** {gradio_interface.predict} NOT AN EMPTY LIST ***')
    gradio_interface.predict.append(
        lambda model_name_or_uuid, *args:
        f(model.get_by_name_or_uuid(name_or_uuid=model_name_or_uuid), *args))

    gradio_interface.input_components.insert(
        0,
        Dropdown(choices=model_names_or_uuids,
                 type='value',
                 default=(model_names_or_uuids[0]
                          if issubclass(model, Model)
                          else model.name_or_uuid),
                 label='H1st Model Name or UUID'))

    _gradio_app, _gradio_path_to_local_server, gradio_share_url = \
        gradio_interface.launch(
            inline=False,
            # (bool) - whether to display in the interface inline
            # on python notebooks.

            inbrowser=True,
            # (bool) - whether to automatically launch the interface
            # in a new tab on the default browser.

            share=True,
            # (bool) - whether to create a publicly shareable link
            # from your computer for the interface.

            debug=False,
            # (bool) - if True, and the interface was launched
            # from Google Colab, prints the errors in the cell output.

            auth=None,
            # (Callable, Union[Tuple[str, str], List[Tuple[str, str]]]) -
            # If provided, username and password
            # (or list of username-password tuples)
            # required to access interface.
            # Can also provide function that takes username and password
            # and returns True if valid login.

            auth_message=None,
            # (str) - If provided, HTML message provided on login page.

            private_endpoint=None,
            prevent_thread_lock=False)

    return HttpResponseRedirect(redirect_to=gradio_share_url)


def exec_on_json_input_data(request, model_name_or_uuid: str, json_input_data):
    model = Model.get_by_name_or_uuid(name_or_uuid=model_name_or_uuid)

    json_input_data = json.loads(json_input_data)

    loaded_json_input_data = load_data_set_pointers_as_json(json_input_data)

    json_output_data = model.predict(loaded_json_input_data)

    saved_json_output_data = \
        save_numpy_arrays_and_pandas_dfs_as_data_set_pointers(json_output_data)

    print(f'OUTPUT: {saved_json_output_data}')

    Decision.objects.create(
        input_data=json_input_data,
        model=model,
        model_code={str(model.uuid): getsource(type(model))},
        output_data=saved_json_output_data)

    return JsonResponse(data=saved_json_output_data,
                        encoder=DjangoJSONEncoder,
                        safe=True,
                        json_dumps_params=None)

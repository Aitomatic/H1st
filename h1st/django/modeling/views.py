from inspect import getsource, isclass
import json

from django.core.serializers.json import DjangoJSONEncoder
from django.http.response import HttpResponseRedirect, Http404, JsonResponse

from gradio.interface import Interface
from gradio.inputs import Dropdown

from ..data_mgmt.util import (
    load_data_set_pointers_as_json,
    save_numpy_arrays_and_pandas_dfs_as_data_set_pointers)
from .models import Model
from ..trust_vault.models import Decision


def launch_gradio_ui(request, model_name_or_uuid: str):
    model = Model.get_by_name_or_uuid(name_or_uuid=model_name_or_uuid)

    gradio_interface = model.gradio_ui

    if gradio_interface is NotImplemented:
        return Http404(f'{type(model).__name__} does not implement Gradio UI')

    else:
        assert isinstance(gradio_interface, Interface), \
            TypeError(f'*** {gradio_interface} NOT A GRADIO INTERFACE ***')

        model_names_or_uuids = model.names_or_uuids

        f = gradio_interface.predict.pop()
        assert not gradio_interface.predict, \
            f'*** {gradio_interface.predict} NOT EMPTY LIST ***'
        gradio_interface.predict.append(
            lambda model_name_or_uuid, *args:
            f(model.get_by_name_or_uuid(name_or_uuid=model_name_or_uuid),
              *args)
        )

        gradio_interface.input_components.insert(
            0,
            Dropdown(choices=model_names_or_uuids,
                     type='value',
                     default=(model_names_or_uuids[0]
                              if isclass(model)
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

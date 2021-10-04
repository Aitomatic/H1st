from inspect import getsource
import json

from django.core.serializers.json import DjangoJSONEncoder
from django.http.response import HttpResponseRedirect, Http404, JsonResponse

from gradio.interface import Interface

from ..data.util import \
    load_data_set_pointers_as_json, \
    save_numpy_arrays_and_pandas_dfs_as_data_set_pointers
from .models import H1stModel
from ..trust.models import Decision


def exec_on_json_input_data(request, model_name_or_uuid: str, json_input_data):
    model = H1stModel.get_by_name_or_uuid(name_or_uuid=model_name_or_uuid)

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


def launch_gradio_ui(request, model_name_or_uuid: str):
    model = H1stModel.get_by_name_or_uuid(name_or_uuid=model_name_or_uuid)

    interface = model.gradio_ui

    if interface is NotImplemented:
        return Http404(f'{type(model).__name__} does not implement Gradio UI')

    else:
        assert isinstance(interface, Interface), \
            TypeError(f'*** {interface} NOT A GRADIO INTERFACE ***')

        _, _, share_url = \
            interface.launch(
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

        return HttpResponseRedirect(redirect_to=share_url)

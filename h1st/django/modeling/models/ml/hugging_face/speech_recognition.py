__all__ = ('PreTrainedHuggingFaceSpeechRecognizer',
           'H1stPreTrainedHuggingFaceSpeechRecognizer')


from typing import Sequence, Union

from gradio.interface import Interface
from gradio.inputs import Audio as AudioInputComponent
from gradio.outputs import Textbox as TextboxOutputComponent

import numpy

from .....util import PGSQL_IDENTIFIER_MAX_LEN, enable_dict_io
from ....apps import H1stAIModelingModuleConfig
from .base import PreTrainedHuggingFaceTransformer


SpeechRecognitionInputType = Union[numpy.ndarray, str]


class PreTrainedHuggingFaceSpeechRecognizer(PreTrainedHuggingFaceTransformer):
    class Meta(PreTrainedHuggingFaceTransformer.Meta):
        verbose_name = 'Pre-Trained Hugging Face Speech Recognizer'
        verbose_name_plural = 'Pre-Trained Hugging Face Speech Recognizers'

        db_table = (f'{H1stAIModelingModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = \
            'h1st_pretrained_hugging_face_speech_recognizers'

    @enable_dict_io
    def predict(self,
                speech_or_speeches:
                    Union[SpeechRecognitionInputType,
                          Sequence[SpeechRecognitionInputType]]) \
            -> Union[str, list[str]]:
        single_speech = isinstance(speech_or_speeches, (numpy.ndarray, str))

        if not (single_speech or isinstance(speech_or_speeches, list)):
            speech_or_speeches = list(speech_or_speeches)

        self.load()

        output = self.native_model_obj(speech_or_speeches)

        return (output['text']
                if single_speech
                else [result['text'] for result in output])

    @property
    def gradio_ui(self) -> Interface:
        def _predict(sampling_rate_and_double_channel_audio_array:
                     tuple[int, numpy.ndarray]) -> str:
            sampling_rate, double_channel_audio_array = \
                sampling_rate_and_double_channel_audio_array

            return self.predict(double_channel_audio_array[:, 0]
                                .astype(numpy.float32))

        return Interface(
            fn=_predict,
            # (Callable) - the function to wrap an interface around.

            inputs=AudioInputComponent(source='microphone',
                                       type='numpy',
                                       label='Recorded Speech to Transcribe',
                                       optional=False),
            # (Union[str, List[Union[str, InputComponent]]]) -
            # a single Gradio input component,
            # or list of Gradio input components.
            # Components can either be passed as instantiated objects,
            # or referred to by their string shortcuts.
            # The number of input components should match
            # the number of parameters in fn.

            outputs=TextboxOutputComponent(type='str',
                                           label='Transcribe Text'),
            # (Union[str, List[Union[str, OutputComponent]]]) -
            # a single Gradio output component,
            # or list of Gradio output components.
            # Components can either be passed as instantiated objects,
            # or referred to by their string shortcuts.
            # The number of output components should match
            # the number of values returned by fn.

            verbose=True,
            # (bool) - whether to print detailed information during launch.

            examples=None,
            # (Union[List[List[Any]], str]) - sample inputs for the function;
            # if provided, appears below the UI components and can be used
            # to populate the interface.
            # Should be nested list, in which the outer list consists of
            # samples and each inner list consists of an input
            # corresponding to each input component.
            # A string path to a directory of examples can also be provided.
            # If there are multiple input components and a directory
            # is provided, a log.csv file must be present in the directory
            # to link corresponding inputs.

            examples_per_page=10,
            # (int) - If examples are provided, how many to display per page.

            live=False,
            # (bool) - should the interface automatically reload on change?

            layout='unaligned',
            # (str) - Layout of input and output panels.
            # - "horizontal" arranges them as two columns of equal height,
            # - "unaligned" arranges them as two columns of unequal height, and
            # - "vertical" arranges them vertically.

            show_input=True,
            show_output=True,

            capture_session=False,
            # (bool) - if True, captures the default graph and session
            # (needed for Tensorflow 1.x)

            interpretation='default',
            # (Union[Callable, str]) - function that provides interpretation
            # explaining prediction output.
            # Pass "default" to use built-in interpreter.

            num_shap=2.0,
            # (float) - a multiplier that determines how many examples
            # are computed for shap-based interpretation.
            # Increasing this value will increase shap runtime,
            # but improve results.

            theme='default',
            # (str) - Theme to use - one of
            # - "default",
            # - "compact",
            # - "huggingface", or
            # - "darkhuggingface".

            repeat_outputs_per_model=True,

            title=self.name,
            # (str) - a title for the interface;
            # if provided, appears above the input and output components.

            description='A pre-trained Hugging Face model to recognize speech',
            # (str) - a description for the interface;
            # if provided, appears above the input and output components.

            article=None,
            # (str) - an expanded article explaining the interface;
            # if provided, appears below the input and output components.
            # Accepts Markdown and HTML content.

            thumbnail=None,
            # (str) - path to image or src to use as display picture for models
            # listed in gradio.app/hub

            css=None,
            # (str) - custom css or path to custom css file
            # to use with interface.

            server_port=None,
            # (int) - will start gradio app on this port (if available)

            # server_name=networking.LOCALHOST_NAME,
            # (str) - to make app accessible on local network set to "0.0.0.0".

            height=500,
            width=900,

            allow_screenshot=True,
            # (bool) - if False, users will not see a button
            # to take a screenshot of the interface.

            allow_flagging=True,
            # (bool) - if False, users will not see a button
            # to flag an input and output.

            flagging_options=None,
            # (List[str]) - if not None, provides options a user must select
            # when flagging.

            encrypt=False,
            # (bool) - If True, flagged data will be encrypted
            # by key provided by creator at launch

            show_tips=False,
            # (bool) - if True, will occasionally show tips
            # about new Gradio features

            embedding=None,

            flagging_dir='flagged',
            # (str) - what to name the dir where flagged data is stored.

            analytics_enabled=True,

            enable_queue=False,
            # (bool) - if True, inference requests will be served through
            # a queue instead of with parallel threads.
            # Required for longer inference times (> 1min) to prevent timeout.
        )


# alias
H1stPreTrainedHuggingFaceSpeechRecognizer = \
    PreTrainedHuggingFaceSpeechRecognizer

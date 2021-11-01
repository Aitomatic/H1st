from typing import Dict, List, Sequence   # TODO: Py3.9: use generics
from typing import Optional, Union

from django.utils.functional import classproperty

from gradio.interface import Interface
from gradio.inputs import (Textbox as TextboxInputComponent,
                           Dataframe as DataframeInputComponent,
                           Slider as SliderInputComponent)
from gradio.outputs import Label as LabelOutputComponent

from .....util import PGSQL_IDENTIFIER_MAX_LEN, enable_dict_io
from ....apps import H1stAIModelingModuleConfig
from .base import PreTrainedHuggingFaceTransformer


__all__: Sequence[str] = ('PreTrainedHuggingFaceMaskFiller',
                          'H1stPreTrainedHuggingFaceMaskFiller')


MaskFillingInputType = str
MaskFillingOutputType = Dict[str, float]


class PreTrainedHuggingFaceMaskFiller(PreTrainedHuggingFaceTransformer):
    class Meta(PreTrainedHuggingFaceTransformer.Meta):
        verbose_name: str = 'Pre-Trained Hugging Face Mask Filler'
        verbose_name_plural: str = 'Pre-Trained Hugging Face Mask Fillers'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_pretrained_hugging_face_mask_fillers'

    @enable_dict_io
    def predict(self,
                text_or_texts: Union[MaskFillingInputType,
                                     Sequence[MaskFillingInputType]],
                targets: Optional[List[str]] = None,
                n_labels: int = 5) \
            -> Union[MaskFillingOutputType, List[MaskFillingOutputType]]:
        single_text: bool = isinstance(text_or_texts, str)

        if not (single_text or isinstance(text_or_texts, list)):
            text_or_texts: List[MaskFillingInputType] = list(text_or_texts)

        self.load()

        output = self.native_model_obj(text_or_texts,
                                       targets=targets,
                                       top_k=n_labels)

        return ({i['token_str']: i['score'] for i in output}
                if single_text or (len(text_or_texts) == 1)
                else [{i['token_str']: i['score'] for i in result}
                      for result in output])

    @classproperty
    def gradio_ui(cls) -> Interface:
        def _predict(self,
                     text: str,
                     targets: Optional[List[str]] = None,
                     n_labels: int = 5) -> Dict[str, float]:
            if targets:
                targets: List[str] = [s for s in targets if s]

            return cls.predict(self,
                               text_or_texts=text,
                               targets=targets if targets else None,
                               n_labels=n_labels)

        return Interface(
            fn=_predict,
            # (Callable) - the function to wrap an interface around.

            inputs=[TextboxInputComponent(lines=10,
                                          placeholder='Text to Generate from',
                                          default='',
                                          numeric=False,
                                          type='str',
                                          label='Text to Generate from'),

                    DataframeInputComponent(headers=None,
                                            row_count=10,
                                            col_count=1,
                                            datatype='str',
                                            col_width=100,
                                            default=None,
                                            type='array',
                                            label='Targets'),

                    SliderInputComponent(minimum=3, maximum=10, step=1,
                                         default=5,
                                         label='No. of Labels to Return')],
            # (Union[str, List[Union[str, InputComponent]]]) -
            # a single Gradio input component,
            # or list of Gradio input components.
            # Components can either be passed as instantiated objects,
            # or referred to by their string shortcuts.
            # The number of input components should match
            # the number of parameters in fn.

            outputs=LabelOutputComponent(num_top_classes=10,
                                         type='auto',
                                         label='Generated Text'),
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
            # - "huggingface",
            # - "grass",
            # - "peach".
            # Add "dark" prefix, e.g. "darkpeach" or "darkdefault"
            # for darktheme.

            repeat_outputs_per_model=True,

            title=cls._meta.verbose_name,
            # (str) - a title for the interface;
            # if provided, appears above the input and output components.

            description=('A pre-trained Hugging Face model for mask-filling'),
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

            allow_flagging=False,
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

            flagging_dir='flagged',
            # (str) - what to name the dir where flagged data is stored.

            analytics_enabled=True,

            enable_queue=False,
            # (bool) - if True, inference requests will be served through
            # a queue instead of with parallel threads.
            # Required for longer inference times (> 1min) to prevent timeout.
        )


# alias
H1stPreTrainedHuggingFaceMaskFiller = PreTrainedHuggingFaceMaskFiller

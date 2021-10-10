from .base import PreTrainedHuggingFaceTransformer

from typing import Sequence, Union

from gradio.interface import Interface
from gradio.inputs import (Image as ImageInputComponent,
                           Slider as SliderInputComponent)
from gradio.outputs import Label as LabelOutputComponent

from PIL.Image import Image

from .....util import PGSQL_IDENTIFIER_MAX_LEN, enable_dict_io
from ....apps import H1stAIModelingModuleConfig


InputImageDataType = Union[str, Image]
OutputImageClassificationType = dict[str, float]


class PreTrainedHuggingFaceImageClassifier(PreTrainedHuggingFaceTransformer):
    class Meta(PreTrainedHuggingFaceTransformer.Meta):
        verbose_name = 'Pre-Trained Hugging Face Image Classifier'
        verbose_name_plural = 'Pre-Trained Hugging Face Image Classifiers'

        db_table = (f'{H1stAIModelingModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_pretrained_hugging_face_image_classifiers'

    @enable_dict_io
    def predict(self,
                image_or_images: Union[InputImageDataType,
                                       Sequence[InputImageDataType]],
                n_labels: int = 5) \
            -> Union[OutputImageClassificationType,
                     Sequence[OutputImageClassificationType]]:
        self.load()

        return {i['label']: i['score']
                for i in self._native_obj(images=image_or_images,
                                          top_k=n_labels)}

    @property
    def gradio_ui(self) -> Interface:
        return Interface(
            fn=self.predict,
            # (Callable) - the function to wrap an interface around.

            inputs=[ImageInputComponent(shape=None,
                                        image_mode='RGB',
                                        invert_colors=False,
                                        source='upload',
                                        tool='editor',
                                        type='pil',
                                        label='Upload an Image to Classify',
                                        optional=False),

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
                                         label='Likely ImageNet Classes'),
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

            description='A pre-trained Hugging Face model to classify images',
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
H1stPreTrainedHuggingFaceImageClassifier = PreTrainedHuggingFaceImageClassifier

__all__ = 'Model', 'H1stModel'


from collections.abc import Generator
from json.decoder import JSONDecoder

from django.core.serializers.json import DjangoJSONEncoder
from django.db.models.fields.json import JSONField
from django.utils.functional import classproperty

from polymorphic.base import PolymorphicModelBase
from polymorphic.models import PolymorphicModel

from django_plotly_dash import DjangoDash

from gradio.interface import Interface
from gradio.outputs import JSON as JSONOutputComponent

from h1st.model.model import Model as CoreH1stModel

from ...util import PGSQL_IDENTIFIER_MAX_LEN, fullqualname
from ...util.models import _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps
from ..apps import H1stAIModelingModuleConfig


class Model(PolymorphicModel,
            _ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps,
            CoreH1stModel):
    params = \
        JSONField(
            verbose_name='Model Parameters',
            help_text='Model Parameters',

            encoder=DjangoJSONEncoder,
            decoder=JSONDecoder,

            null=True,
            blank=True,
            choices=None,
            db_column=None,
            db_index=False,
            db_tablespace=None,
            default=None,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=False,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    class Meta(_ModelWithUUIDPKAndOptionalUniqueNameAndTimestamps.Meta):
        verbose_name = 'H1st Model'
        verbose_name_plural = 'H1st Models'

        db_table = (f'{H1stAIModelingModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_models'

    @classproperty
    def _subclasses(cls) -> Generator[PolymorphicModelBase, None, None]:
        for s in cls.__subclasses__():
            yield s
            yield from s._subclasses

    @classproperty
    def subclasses(cls) -> list[PolymorphicModelBase]:
        return list(cls._subclasses)

    @classproperty
    def subclass_names(cls) -> list[str]:
        return [s.__name__ for s in cls._subclasses]

    @classproperty
    def subclasses_by_name(cls) -> dict[str, PolymorphicModelBase]:
        return {s.__name__: s for s in cls._subclasses}

    @classproperty
    def subclass_full_qual_names(cls) -> list[str]:
        return [fullqualname(s) for s in cls._subclasses]

    @classproperty
    def subclasses_by_full_qual_name(cls) -> dict[str, PolymorphicModelBase]:
        return {fullqualname(s): s for s in cls._subclasses}

    @classproperty
    def dash_ui(cls) -> DjangoDash:
        return NotImplemented

    @classproperty
    def gradio_ui(cls) -> Interface:
        return Interface(
            fn=cls.predict,
            # (Callable) - the function to wrap an interface around.

            inputs=[],
            # (Union[str, List[Union[str, InputComponent]]]) -
            # a single Gradio input component,
            # or list of Gradio input components.
            # Components can either be passed as instantiated objects,
            # or referred to by their string shortcuts.
            # The number of input components should match
            # the number of parameters in fn.

            outputs=JSONOutputComponent(label='H1st Model Output'),
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

            description='A H1st Model',
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

    @classproperty
    def streamlit_ui(cls):
        return NotImplemented


# aliases
H1stModel = Model

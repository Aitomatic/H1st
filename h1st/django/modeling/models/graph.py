from abc import abstractmethod
from collections.abc import Sequence

from django.utils.functional import classproperty

from h1st.h1flow.h1flow import Graph as CoreH1stWorkflow

from ...util import PGSQL_IDENTIFIER_MAX_LEN, enable_dict_io
from ..apps import H1stAIModelingModuleConfig

from .base import H1stModel


__all__: Sequence[str] = 'Graph', 'H1stGraph', 'Workflow', 'H1stWorkflow'


class Workflow(H1stModel, CoreH1stWorkflow):
    class Meta(H1stModel.Meta):
        verbose_name: str = 'H1st Workflow'
        verbose_name_plural: str = 'H1st Workflows'

        db_table: str = (f'{H1stAIModelingModuleConfig.label}_'
                         f"{__qualname__.split(sep='.', maxsplit=1)[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name: str = 'h1st_workflows'

    @classproperty
    @abstractmethod
    def args(cls) -> Sequence[str]:
        raise NotImplementedError

    @enable_dict_io
    def predict(self, *args, **kwargs):
        return CoreH1stWorkflow.predict(self, *args, **kwargs)


# alias
H1stGraph = Graph = H1stWorkflow = Workflow

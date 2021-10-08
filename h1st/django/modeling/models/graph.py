from abc import abstractmethod

from h1st.h1flow.h1flow import Graph as CoreH1stWorkflow

from ...util import PGSQL_IDENTIFIER_MAX_LEN, enable_dict_io
from ..apps import H1stAIModelingModuleConfig

from .base import H1stModel


class Workflow(H1stModel, CoreH1stWorkflow):
    class Meta(H1stModel.Meta):
        verbose_name = 'H1st Workflow'
        verbose_name_plural = 'H1st Workflows'

        db_table = (f'{H1stAIModelingModuleConfig.label}_'
                    f"{__qualname__.split('.')[0]}")
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_workflows'

    @property
    @abstractmethod
    def args(self) -> tuple[str]:
        raise NotImplementedError

    @enable_dict_io
    def predict(self, *args, **kwargs):
        return CoreH1stWorkflow.predict(self, *args, **kwargs)


# alias
H1stGraph = Graph = H1stWorkflow = Workflow

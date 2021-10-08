from ....util import PGSQL_IDENTIFIER_MAX_LEN
from ...apps import H1stModelModuleConfig
from .base import H1stMLModel


class SKLModel(H1stMLModel):
    class Meta(H1stMLModel.Meta):
        verbose_name = 'H1st SciKit-Learn Model'
        verbose_name_plural = 'H1st SciKit-Learn Models'

        db_table = \
            f"{H1stModelModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_skl_models'

    # by default, serialize model object by JobLib/Pickle
    def serialize(self):
        assert self.obj, ValueError(f'*** MODEL OBJECT {self.obj} INVALID ***')

        with TemporaryFile() as f:
            joblib.dump(
                value=self.obj,
                filename=f,
                compress=0,
                protocol=3,   # default protocol in Python 3.0–3.7
                cache_size=None)
            f.seek(0)
            self.artifact = f.read()

    # by default, deserialize model object by JobLib/Pickle
    def deserialize(self):
        with TemporaryFile() as f:
            f.write(self.artifact)
            f.seek(0)
            self.obj = joblib.load(filename=f, mmap_mode=None)

        assert self.obj, ValueError(f'*** MODEL OBJECT {self.obj} INVALID ***')

    def save(self, *args, **kwargs):
        self.serialize()
        super().save(*args, **kwargs)


# alias
H1stSKLModel = SKLModel

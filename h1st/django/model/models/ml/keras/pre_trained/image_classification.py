from django.db.models.fields import CharField
import numpy
from PIL import Image, ImageOps
from tensorflow.python.keras.applications import imagenet_utils

from ......util import PGSQL_IDENTIFIER_MAX_LEN
from .....apps import H1stModelModuleConfig
from ...base import H1stPyLoadablePreTrainedMLModel


class PreTrainedKerasImageNetClassifier(H1stPyLoadablePreTrainedMLModel):
    preprocessor_module_and_qualname = \
        CharField(
            verbose_name='Preprocessor (module.qualname)',
            help_text='Preprocessor (module.qualname)',

            max_length=255,

            null=False,
            blank=False,
            choices=None,
            db_column=None,
            db_index=True,
            db_tablespace=None,
            default=None,
            editable=True,
            # error_messages=None,
            primary_key=False,
            unique=False,
            unique_for_date=None, unique_for_month=None, unique_for_year=None,
            # validators=None
        )

    class Meta(H1stPyLoadablePreTrainedMLModel.Meta):
        verbose_name = 'Pre-Trained Keras ImageNet Classifier'
        verbose_name_plural = 'Pre-Trained Keras ImageNet Classifiers'

        db_table = \
            f"{H1stModelModuleConfig.label}_{__qualname__.split('.')[0]}"
        assert len(db_table) <= PGSQL_IDENTIFIER_MAX_LEN, \
            ValueError(f'*** "{db_table}" DB TABLE NAME TOO LONG ***')

        default_related_name = 'h1st_pretrained_keras_imagenet_classifiers'

    def predict(self,
                image_file_path: str, n_labels: int = 5) -> dict[str, float]:
        self.load()

        # load image from file
        image = Image.open(fp=image_file_path, mode='r', formats=None)

        # fit image to size model expects
        img_dim_size = self.params['img_dim_size']

        fitted_image = ImageOps.fit(image=image,
                                    size=(img_dim_size, img_dim_size),
                                    method=Image.LANCZOS,   # BICUBIC
                                    bleed=0,
                                    centering=(0.5, 0.5))

        # convert fitted image to NumPy array
        fitted_image_array = numpy.asarray(fitted_image, dtype=int, order=None)

        # make a batch of 1 array
        fitted_image_batch_array = numpy.expand_dims(fitted_image_array,
                                                     axis=0)

        # preprocess
        preprocessed_fitted_image_batch_array = \
            self.preprocessor(fitted_image_batch_array)

        # predict
        probabilities = \
            self._native_model_obj.predict(
                x=preprocessed_fitted_image_batch_array)

        # return dict
        return imagenet_utils.decode_predictions(preds=probabilities,
                                                 top=n_labels)


# alias
H1stPreTrainedKerasImageNetClassifier = PreTrainedKerasImageNetClassifier

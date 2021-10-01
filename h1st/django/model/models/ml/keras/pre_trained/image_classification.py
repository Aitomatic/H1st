from typing import Sequence, Union

from django.db.models.fields import CharField
import numpy
from PIL import Image, ImageOps
from tensorflow.python.keras.applications import imagenet_utils

from ......util import PGSQL_IDENTIFIER_MAX_LEN, import_obj
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

    @property
    def preprocessor(self) -> callable:
        return import_obj(self.preprocessor_module_and_qualname)

    def predict(self,
                image_file_or_files: Union[str, Sequence[str]],
                n_labels: int = 5) -> dict[str, float]:
        single_img = isinstance(image_file_or_files, str)

        img_file_paths = ([image_file_or_files]
                          if single_img
                          else image_file_or_files)

        # construct 4D array of images' data fitted into standardized size
        fitted_img_arrs = []
        img_dim_size = self.params['img_dim_size']
        for img_file_path in img_file_paths:
            # load image from file
            img = Image.open(fp=img_file_path, mode='r', formats=None)
            # fit image to size model expects
            fitted_img = ImageOps.fit(image=img,
                                      size=(img_dim_size, img_dim_size),
                                      method=Image.LANCZOS,
                                      bleed=0,
                                      centering=(0.5, 0.5))
            # convert fitted image to batch of 1 3D NumPy array
            fitted_img_arrs.append(numpy.expand_dims(numpy.asarray(fitted_img,
                                                                   dtype=int,
                                                                   order=None),
                                                     axis=0))
        fitted_img_batch_arr = numpy.vstack(fitted_img_arrs)

        # preprocess
        preprocessed_fitted_img_batch_arr = \
            self.preprocessor(fitted_img_batch_arr)

        # load native model object & predict
        self.load()

        pred_prob_arr = \
            self._native_obj.predict(
                x=preprocessed_fitted_img_batch_arr)

        # decode predictions & return
        decoded_preds = [{tup[1]: tup[2] for tup in decoded_pred}
                         for decoded_pred in
                         imagenet_utils.decode_predictions(
                             preds=pred_prob_arr,
                             top=n_labels)]

        return (decoded_preds[0] if single_img else decoded_preds)


# alias
H1stPreTrainedKerasImageNetClassifier = PreTrainedKerasImageNetClassifier

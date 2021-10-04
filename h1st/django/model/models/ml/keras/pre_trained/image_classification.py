from typing import Sequence, Tuple, Union

from django.db.models.fields import CharField

from gradio.inputs import Image as ImageInputComponent
from gradio.outputs import Label as LabelOutputComponent

import numpy
from PIL import Image, ImageOps
from tensorflow.python.keras.applications import imagenet_utils

from ......util import PGSQL_IDENTIFIER_MAX_LEN, import_obj
from .....apps import H1stModelModuleConfig
from ...base import H1stPyLoadablePreTrainedMLModel


InputImageDataType = Union[str, Image.Image, numpy.ndarray]
OutputImageClassificationType = dict[str, float]


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
    def img_dim_size(self):
        return self.params['img_dim_size']

    def image_to_4d_array(self, image: InputImageDataType) -> numpy.ndarray:
        # if string file path, then load from file
        if isinstance(image, str):
            image = Image.open(fp=image, mode='r', formats=None)

        # if PIL.Image.Image instance,
        # then fit to size model expects and then convert to NumPy array
        if isinstance(image, Image.Image):
            img_dim_size = self.img_dim_size
            image = numpy.asarray(ImageOps.fit(image=image,
                                               size=(img_dim_size,
                                                     img_dim_size),
                                               method=Image.LANCZOS,
                                               bleed=0,
                                               centering=(0.5, 0.5)),
                                  dtype=int,
                                  order=None)

        assert isinstance(image, numpy.ndarray), \
            TypeError(f'*** {image} not a NumPy Array ***')

        # convert to batch of 1 3D NumPy array
        return numpy.expand_dims(image, axis=0)

    @property
    def preprocessor(self) -> callable:
        return import_obj(self.preprocessor_module_and_qualname)

    def predict(self,
                image_or_images: Union[InputImageDataType,
                                       Sequence[InputImageDataType]],
                n_labels: int = 5) \
            -> Union[OutputImageClassificationType,
                     Sequence[OutputImageClassificationType]]:
        single_img = isinstance(image_or_images, (str,
                                                  Image.Image,
                                                  numpy.ndarray))

        imgs = [image_or_images] if single_img else image_or_images

        # construct 4D array of images' data fitted into standardized size
        fitted_img_batch_arr = numpy.vstack([self.image_to_4d_array(image=img)
                                             for img in imgs])

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

    @property
    def gradio_ui(self) -> Tuple[ImageInputComponent, LabelOutputComponent]:
        img_dim_size = self.img_dim_size

        return (ImageInputComponent(shape=(img_dim_size, img_dim_size),
                                    image_mode='RGB',
                                    invert_colors=False,
                                    source='upload',
                                    tool='editor',
                                    type='numpy',
                                    label='Upload an Image to Classify',
                                    optional=False),

                LabelOutputComponent(num_top_classes=5,
                                     type='auto',
                                     label=f'Predictions by {self}'))


# alias
H1stPreTrainedKerasImageNetClassifier = PreTrainedKerasImageNetClassifier

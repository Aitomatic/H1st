from abc import abstractmethod

import numpy
from PIL import Image, ImageOps

from tensorflow.keras.applications.imagenet_utils import decode_predictions

from h1st.django.model.api import H1stKerasModel


class KerasImageNetClassifierMixIn:
    @property
    @abstractmethod
    def preprocessor(self) -> callable:
        raise NotImplementedError

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
        return decode_predictions(preds=probabilities, top=n_labels)


class AbstractKerasImageNetClassifier(KerasImageNetClassifierMixIn,
                                      H1stKerasModel):
    class Meta:
        abstract = True

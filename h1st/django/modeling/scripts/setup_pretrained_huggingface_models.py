from transformers.pipelines import pipeline

from ...util import fullqualname
from ..models import PreTrainedHuggingFaceImageClassifier


def run():
    print(
        PreTrainedHuggingFaceImageClassifier.objects.update_or_create(
            name='PreTrained-HuggingFace-Image-Classifier',
            defaults=dict(
                py_loader_module_and_qualname=fullqualname(pipeline),
                artifact_global_url=None,
                artifact_local_path=None,
                params=dict(__init__=dict(task='image-classification',
                                          model=None,
                                          config=None,
                                          tokenizer=None,
                                          feature_extractor=None,
                                          framework=None,
                                          revision=None,
                                          use_fast=True,
                                          use_auth_token=None,
                                          model_kwargs={}))))[0])

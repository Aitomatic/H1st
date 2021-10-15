from transformers.pipelines import pipeline

from ...util import fullqualname
from ..models import (
    PreTrainedHuggingFaceImageClassifier,
    PreTrainedHuggingFaceTextClassifier,
    PreTrainedHuggingFaceTokenClassifier,
)


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

    print(
        PreTrainedHuggingFaceTextClassifier.objects.update_or_create(
            name='PreTrained-HuggingFace-Text-Classifier',
            defaults=dict(
                py_loader_module_and_qualname=fullqualname(pipeline),
                artifact_global_url=None,
                artifact_local_path=None,
                params=dict(__init__=dict(task='text-classification',
                                          model=None,
                                          config=None,
                                          tokenizer=None,
                                          feature_extractor=None,
                                          framework=None,
                                          revision=None,
                                          use_fast=True,
                                          use_auth_token=None,
                                          model_kwargs={}))))[0])
    print(
        PreTrainedHuggingFaceTextClassifier.objects.update_or_create(
            name='PreTrained-HuggingFace-Sentiment-Analyzer',
            defaults=dict(
                py_loader_module_and_qualname=fullqualname(pipeline),
                artifact_global_url=None,
                artifact_local_path=None,
                params=dict(__init__=dict(task='sentiment-analysis',
                                          model=None,
                                          config=None,
                                          tokenizer=None,
                                          feature_extractor=None,
                                          framework=None,
                                          revision=None,
                                          use_fast=True,
                                          use_auth_token=None,
                                          model_kwargs={}))))[0])

    print(
        PreTrainedHuggingFaceTokenClassifier.objects.update_or_create(
            name='PreTrained-HuggingFace-Token-Classifier',
            defaults=dict(
                py_loader_module_and_qualname=fullqualname(pipeline),
                artifact_global_url=None,
                artifact_local_path=None,
                params=dict(__init__=dict(task='token-classification',
                                          model=None,
                                          config=None,
                                          tokenizer=None,
                                          feature_extractor=None,
                                          framework=None,
                                          revision=None,
                                          use_fast=True,
                                          use_auth_token=None,
                                          model_kwargs={}))))[0])
    print(
        PreTrainedHuggingFaceTokenClassifier.objects.update_or_create(
            name='PreTrained-HuggingFace-Named-Entity-Recognizer',
            defaults=dict(
                py_loader_module_and_qualname=fullqualname(pipeline),
                artifact_global_url=None,
                artifact_local_path=None,
                params=dict(__init__=dict(task='ner',
                                          model=None,
                                          config=None,
                                          tokenizer=None,
                                          feature_extractor=None,
                                          framework=None,
                                          revision=None,
                                          use_fast=True,
                                          use_auth_token=None,
                                          model_kwargs={}))))[0])

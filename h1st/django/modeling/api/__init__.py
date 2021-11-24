"""H1st Django Modeling public API."""


import sys

from ..models import (
    Model, H1stModel,

    KnowledgeModel,
    BooleanLogicKnowledgeModel,

    CloudServiceModel, H1stCloudServiceModel,

    GoogleCloudTranslationServiceModel, H1stGoogleCloudTranslationServiceModel,
    GoogleTranslateServiceModel, H1stGoogleTranslateServiceModel,

    MLModel, H1stMLModel,
    PyLoadablePreTrainedMLModel, H1stPyLoadablePreTrainedMLModel,

    SKLModel, H1stSKLModel,

    TFModel, H1stTFModel,
    PreTrainedTFHubTransformer, H1stPreTrainedTFHubTransformer,

    KerasModel, H1stKerasModel,
    PreTrainedKerasImageNetClassifier, H1stPreTrainedKerasImageNetClassifier,

    TorchModel, H1stTorchModel,
    PreTrainedTorchVisionImageNetClassifier,
    H1stPreTrainedTorchVisionImageNetClassifier,

    PreTrainedHuggingFaceTransformer, H1stPreTrainedHuggingFaceTransformer,

    PreTrainedHuggingFaceAudioClassifier,
    H1stPreTrainedHuggingFaceAudioClassifier,

    PreTrainedHuggingFaceImageClassifier,
    H1stPreTrainedHuggingFaceImageClassifier,

    PreTrainedHuggingFaceMaskFiller,
    H1stPreTrainedHuggingFaceMaskFiller,

    PreTrainedHuggingFaceObjectDetector,
    H1stPreTrainedHuggingFaceObjectDetector,

    PreTrainedHuggingFaceQuestionAnswerer,
    H1stPreTrainedHuggingFaceQuestionAnswerer,

    PreTrainedHuggingFaceSpeechRecognizer,
    H1stPreTrainedHuggingFaceSpeechRecognizer,

    PreTrainedHuggingFaceTableQuestionAnswerer,
    H1stPreTrainedHuggingFaceTableQuestionAnswerer,

    PreTrainedHuggingFaceTextClassifier,
    H1stPreTrainedHuggingFaceTextClassifier,

    PreTrainedHuggingFaceTextGenerator,
    H1stPreTrainedHuggingFaceTextGenerator,

    PreTrainedHuggingFaceText2TextGenerator,
    H1stPreTrainedHuggingFaceText2TextGenerator,

    PreTrainedHuggingFaceTextSummarizer,
    H1stPreTrainedHuggingFaceTextSummarizer,

    PreTrainedHuggingFaceTokenClassifier,
    H1stPreTrainedHuggingFaceTokenClassifier,

    PreTrainedHuggingFaceTranslator,
    H1stPreTrainedHuggingFaceTranslator,

    PreTrainedHuggingFaceZeroShotClassifier,
    H1stPreTrainedHuggingFaceZeroShotClassifier,

    Graph, H1stGraph, Workflow, H1stWorkflow)

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = (
    'Model', 'H1stModel',

    'KnowledgeModel',
    'BooleanLogicKnowledgeModel',

    'CloudServiceModel', 'H1stCloudServiceModel',

    'GoogleCloudTranslationServiceModel',
    'H1stGoogleCloudTranslationServiceModel',
    'GoogleTranslateServiceModel', 'H1stGoogleTranslateServiceModel',

    'MLModel', 'H1stMLModel',

    'PyLoadablePreTrainedMLModel', 'H1stPyLoadablePreTrainedMLModel',

    'SKLModel', 'H1stSKLModel',

    'TFModel', 'H1stTFModel',
    'PreTrainedTFHubTransformer', 'H1stPreTrainedTFHubTransformer',

    'KerasModel', 'H1stKerasModel',
    'PreTrainedKerasImageNetClassifier',
    'H1stPreTrainedKerasImageNetClassifier',

    'TorchModel', 'H1stTorchModel',
    'PreTrainedTorchVisionImageNetClassifier',
    'H1stPreTrainedTorchVisionImageNetClassifier',

    'PreTrainedHuggingFaceTransformer',
    'H1stPreTrainedHuggingFaceTransformer',

    'PreTrainedHuggingFaceAudioClassifier',
    'H1stPreTrainedHuggingFaceAudioClassifier',

    'PreTrainedHuggingFaceImageClassifier',
    'H1stPreTrainedHuggingFaceImageClassifier',

    'PreTrainedHuggingFaceMaskFiller',
    'H1stPreTrainedHuggingFaceMaskFiller',

    'PreTrainedHuggingFaceObjectDetector',
    'H1stPreTrainedHuggingFaceObjectDetector',

    'PreTrainedHuggingFaceQuestionAnswerer',
    'H1stPreTrainedHuggingFaceQuestionAnswerer',

    'PreTrainedHuggingFaceSpeechRecognizer',
    'H1stPreTrainedHuggingFaceSpeechRecognizer',

    'PreTrainedHuggingFaceTableQuestionAnswerer',
    'H1stPreTrainedHuggingFaceTableQuestionAnswerer',

    'PreTrainedHuggingFaceTextClassifier',
    'H1stPreTrainedHuggingFaceTextClassifier',

    'PreTrainedHuggingFaceTextGenerator',
    'H1stPreTrainedHuggingFaceTextGenerator',

    'PreTrainedHuggingFaceText2TextGenerator',
    'H1stPreTrainedHuggingFaceText2TextGenerator',

    'PreTrainedHuggingFaceTextSummarizer',
    'H1stPreTrainedHuggingFaceTextSummarizer',

    'PreTrainedHuggingFaceTokenClassifier',
    'H1stPreTrainedHuggingFaceTokenClassifier',

    'PreTrainedHuggingFaceTranslator',
    'H1stPreTrainedHuggingFaceTranslator',

    'PreTrainedHuggingFaceZeroShotClassifier',
    'H1stPreTrainedHuggingFaceZeroShotClassifier',

    'Graph', 'H1stGraph', 'Workflow', 'H1stWorkflow')

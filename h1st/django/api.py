"""H1st Django public API."""


import sys

from h1st.django.data_mgmt.api import (
    DataSchema,
    DataSet,
    JSONDataSet,
    NumPyArray,
    PandasDataFrame,
    ParquetDataSet,
    CSVDataSet,
    TFRecordDataSet,
    TextDataSet,
    LiveDataSource,
)
from h1st.django.modeling.api import (
    Model, H1stModel,

    KnowledgeModel,
    BooleanLogicKnowledgeModel,

    CloudServiceModel, H1stCloudServiceModel,

    GoogleCloudTranslationServiceModel,
    H1stGoogleCloudTranslationServiceModel,

    GoogleTranslateServiceModel,
    H1stGoogleTranslateServiceModel,

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

    PreTrainedHuggingFaceTransformer,
    H1stPreTrainedHuggingFaceTransformer,

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

from h1st.django.trust_vault.api import Decision, ModelEvalMetricsSet

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = (
    'DataSchema',
    'DataSet',
    'JSONDataSet',
    'NumPyArray',
    'PandasDataFrame',
    'ParquetDataSet',
    'CSVDataSet',
    'TFRecordDataSet',
    'TextDataSet',
    'LiveDataSource',

    'Model', 'H1stModel',

    'KnowledgeModel',
    'BooleanLogicKnowledgeModel',

    'CloudServiceModel', 'H1stCloudServiceModel',

    'GoogleCloudTranslationServiceModel',
    'H1stGoogleCloudTranslationServiceModel',

    'GoogleTranslateServiceModel',
    'H1stGoogleTranslateServiceModel',

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

    'Graph', 'H1stGraph', 'Workflow', 'H1stWorkflow',

    'Decision', 'ModelEvalMetricsSet',
)

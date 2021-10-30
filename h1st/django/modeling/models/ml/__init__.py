from typing import Sequence   # TODO: Py3.9: use generic collections.abc

from .base import (MLModel, H1stMLModel,
                   PyLoadablePreTrainedMLModel, H1stPyLoadablePreTrainedMLModel
                   )

from .skl import SKLModel, H1stSKLModel

from .tf import (
    TFModel, H1stTFModel,
    PreTrainedTFHubTransformer, H1stPreTrainedTFHubTransformer,
)

from .keras import (
    KerasModel, H1stKerasModel,
    PreTrainedKerasImageNetClassifier, H1stPreTrainedKerasImageNetClassifier,
)

from .torch import (
    TorchModel, H1stTorchModel,
    PreTrainedTorchVisionImageNetClassifier,
    H1stPreTrainedTorchVisionImageNetClassifier,
)

from .hugging_face import (
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
)


__all__: Sequence[str] = (
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

    'PreTrainedHuggingFaceTransformer', 'H1stPreTrainedHuggingFaceTransformer',

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
)

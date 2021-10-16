__all__ = (
    'PreTrainedHuggingFaceTransformer', 'H1stPreTrainedHuggingFaceTransformer',

    'PreTrainedHuggingFaceAudioClassifier',
    'H1stPreTrainedHuggingFaceAudioClassifier',

    'PreTrainedHuggingFaceImageClassifier',
    'H1stPreTrainedHuggingFaceImageClassifier',

    'PreTrainedHuggingFaceObjectDetector',
    'H1stPreTrainedHuggingFaceObjectDetector',

    'PreTrainedHuggingFaceSpeechRecognizer',
    'H1stPreTrainedHuggingFaceSpeechRecognizer',

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
)


from .base import (PreTrainedHuggingFaceTransformer,
                   H1stPreTrainedHuggingFaceTransformer)
from .audio_classification import (PreTrainedHuggingFaceAudioClassifier,
                                   H1stPreTrainedHuggingFaceAudioClassifier)
from .image_classification import (PreTrainedHuggingFaceImageClassifier,
                                   H1stPreTrainedHuggingFaceImageClassifier)
from .object_detection import (PreTrainedHuggingFaceObjectDetector,
                               H1stPreTrainedHuggingFaceObjectDetector)
from .speech_recognition import (PreTrainedHuggingFaceSpeechRecognizer,
                                 H1stPreTrainedHuggingFaceSpeechRecognizer)
from .text_classification import (PreTrainedHuggingFaceTextClassifier,
                                  H1stPreTrainedHuggingFaceTextClassifier)
from .text_generation import (PreTrainedHuggingFaceTextGenerator,
                              H1stPreTrainedHuggingFaceTextGenerator)
from .text2text_generation import (PreTrainedHuggingFaceText2TextGenerator,
                                   H1stPreTrainedHuggingFaceText2TextGenerator)
from .text_summarization import (PreTrainedHuggingFaceTextSummarizer,
                                 H1stPreTrainedHuggingFaceTextSummarizer)
from .token_classification import (PreTrainedHuggingFaceTokenClassifier,
                                   H1stPreTrainedHuggingFaceTokenClassifier)
from .translation import (PreTrainedHuggingFaceTranslator,
                          H1stPreTrainedHuggingFaceTranslator)

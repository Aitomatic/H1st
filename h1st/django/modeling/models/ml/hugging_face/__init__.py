from collections.abc import Sequence

from .base import (PreTrainedHuggingFaceTransformer,
                   H1stPreTrainedHuggingFaceTransformer)
from .audio_classification import (PreTrainedHuggingFaceAudioClassifier,
                                   H1stPreTrainedHuggingFaceAudioClassifier)
from .image_classification import (PreTrainedHuggingFaceImageClassifier,
                                   H1stPreTrainedHuggingFaceImageClassifier)
from .mask_filling import (PreTrainedHuggingFaceMaskFiller,
                           H1stPreTrainedHuggingFaceMaskFiller)
from .object_detection import (PreTrainedHuggingFaceObjectDetector,
                               H1stPreTrainedHuggingFaceObjectDetector)
from .question_answering import (PreTrainedHuggingFaceQuestionAnswerer,
                                 H1stPreTrainedHuggingFaceQuestionAnswerer)
from .speech_recognition import (PreTrainedHuggingFaceSpeechRecognizer,
                                 H1stPreTrainedHuggingFaceSpeechRecognizer)
from .table_question_answering import (
    PreTrainedHuggingFaceTableQuestionAnswerer,
    H1stPreTrainedHuggingFaceTableQuestionAnswerer)
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
from .zero_shot_classification import (
    PreTrainedHuggingFaceZeroShotClassifier,
    H1stPreTrainedHuggingFaceZeroShotClassifier)


__all__: Sequence[str] = (
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

import logging
from typing import Optional
from nemo.collections.tts.g2p.models.fa_ir_persian.phonemizer import PersianPhonemizer
from abc import ABC, abstractmethod
from typing import Optional

# ===============================
# From: sadeghkrmi/NeMo: nemo/collections/tts/g2p/models/base.py
# ===============================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseG2p(ABC):
    def __init__(
        self,
        phoneme_dict=None,
        word_tokenize_func=lambda x: x,
        apply_to_oov_word=None,
        mapping_file: Optional[str] = None,
    ):
        """Abstract class for creating an arbitrary module to convert grapheme words
        to phoneme sequences, leave unchanged, or use apply_to_oov_word.
        Args:
            phoneme_dict: Arbitrary representation of dictionary (phoneme -> grapheme) for known words.
            word_tokenize_func: Function for tokenizing text to words.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
        """
        self.phoneme_dict = phoneme_dict
        self.word_tokenize_func = word_tokenize_func
        self.apply_to_oov_word = apply_to_oov_word
        self.mapping_file = mapping_file
        self.heteronym_model = None  # heteronym classification model

    @abstractmethod
    def __call__(self, text: str) -> str:
        pass
# ===============================
# End: sadeghkrmi/NeMo: nemo/collections/tts/g2p/models/base.py
# ===============================


# ===============================
# From: sadeghkrmi/NeMo: nemo/collections/tts/g2p/models/fa_ir_persian.py
# ===============================
class PersianG2p(BaseG2p):
    def __init__(
        self,
        phoneme_dict=None,
        ignore_ambiguous_words=True,
        heteronyms=None,
        encoding='utf-8',
        phoneme_probability: Optional[float] = None,
        mapping_file: Optional[str] = None,
    ):
        """
        Persian G2P module.
        Args:
            to be defined.
        """
        assert phoneme_dict is not None, "Please set the phoneme_dict path."
        self.phoner = PersianPhonemizer(dictionary_path=phoneme_dict)
        
    def __call__(self, text):
        phoneme_seq = self.phoner.phonemize(text)
        return phoneme_seq

# ===============================
# End: sadeghkrmi/NeMo: nemo/collections/tts/g2p/models/fa_ir_persian.py
# ===============================
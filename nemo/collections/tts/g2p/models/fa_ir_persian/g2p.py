import logging
import re
from typing import Optional, List, Dict, Union, Any
from abc import ABC, abstractmethod

# Placeholder imports - logic assumes these libraries are installed
try:
    from pernorm.normalizer import PersianNormalizer
    from vaguye.phonemizer import PersianPhonemizer
    from zirneshane import HybridZirneshanModel
    from hamnevise import HamneviseModel
except ImportError:
    logging.warning("One or more Persian NLP libraries (pernorm, vaguye, zirneshane, hamnevise) not found.")
    # Define dummy classes to avoid crashing if libraries are missing during static analysis or initial load
    class PersianNormalizer:
        def normalize(self, text): return text
    class PersianPhonemizer:
        def __init__(self, **kwargs): pass
        def phonemize(self, text): return text
    class HybridZirneshanModel:
        @staticmethod
        def load(): return HybridZirneshanModel()
        def predict(self, text): return text
    class HamneviseModel:
        @staticmethod
        def load(): return HamneviseModel(), None
        def disambiguate(self, text, tokenizer): return text, None

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
        to phoneme sequences."""
        self.phoneme_dict = phoneme_dict
        self.word_tokenize_func = word_tokenize_func
        self.apply_to_oov_word = apply_to_oov_word
        self.mapping_file = mapping_file

    @abstractmethod
    def __call__(self, text: str) -> Any:
        pass


class VaguyePipeline:
    def __init__(self, ipa=True, stress=True, chunk_threshold=100):
        """
        Initialize the Persian text processing pipeline.
        
        Args:
            ipa: Whether to use IPA phonemes
            stress: Whether to include stress marks
            chunk_threshold: Maximum number of words before chunking into sentences
        """
        try:
            # Step 1
            self.normalizer = PersianNormalizer()
            # Step 2
            self.phonemizer = PersianPhonemizer(ipa=ipa, stress=stress)
            # Step 3
            self.zirneshane = HybridZirneshanModel.load()
            # Step 4
            self.hamnevise, self.tokenizer = HamneviseModel.load()
            
            self.chunk_threshold = chunk_threshold
            
            # Cache the set of words that need hamnevise disambiguation
            # word2idx belongs to the tokenizer object returned by load()
            self.hamnevise_words = set(self.tokenizer.word2idx.keys())
        except Exception as e:
            logger.error(f"Error initializing VaguyePipeline: {e}")
            raise e
    
    def _split_sentences(self, text):
        """
        Split Persian text into sentences.
        
        Splits on common sentence delimiters: . ! ? ؟ ۔
        Preserves the delimiter with each sentence.
        """
        # Persian sentence delimiters
        pattern = r'([.!?؟۔])\s+'
        
        # Split while keeping delimiters
        parts = re.split(pattern, text)
        
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentence = parts[i] + parts[i + 1]
                sentences.append(sentence.strip())
        
        # Handle last part if no delimiter at end
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        
        return sentences if sentences else [text]
    
    def _should_chunk(self, text):
        """Determine if text should be chunked based on word count."""
        word_count = len(text.split())
        return word_count > self.chunk_threshold
    
    def _needs_hamnevise(self, text):
        """Check if any word in text requires hamnevise disambiguation."""
        words = text.split()
        return any(word in self.hamnevise_words for word in words)
    
    def _process_single(self, text):
        """Process a single chunk of text through the pipeline."""
        output = {}
        
        # Step 1: Normalize
        norm = self.normalizer.normalize(text)
        output["normalized"] = norm
        
        # Step 2: Hamnevise (only if needed)
        if self._needs_hamnevise(norm):
            ham, _ = self.hamnevise.disambiguate(norm, tokenizer=self.tokenizer)
            output["hamnevise"] = ham
        else:
            # Skip hamnevise if no relevant words found
            ham = norm
            output["hamnevise"] = ham
        
        # Step 3: Zirnešān
        zir = self.zirneshane.predict(ham)
        output["zirneshan"] = zir
        
        # Step 4: Phonemize
        ph = self.phonemizer.phonemize(zir)
        output["phonemes"] = ph
        
        return output
    
    def __call__(self, text, return_chunked=False):
        """
        Run full pipeline on input text.
        
        Args:
            text: Input Persian text
            return_chunked: If True, return list of results per sentence when chunked.
                          If False, concatenate results into single output.
        
        Returns:
            processed data
        """
        # Check if we need to chunk
        if not self._should_chunk(text):
            return self._process_single(text)
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Process each sentence
        results = [self._process_single(sent) for sent in sentences]
        
        # Return based on preference
        if return_chunked:
            return results
        
        # Concatenate results
        concatenated = {
            "normalized": " ".join(r["normalized"] for r in results),
            "hamnevise": " ".join(r["hamnevise"] for r in results),
            "zirneshan": " ".join(r["zirneshan"] for r in results),
            "phonemes": " ".join(r["phonemes"] for r in results),
        }
        
        return concatenated


class PersianG2p(BaseG2p):
    def __init__(
        self,
        phoneme_dict=None,
        ignore_ambiguous_words=True,
        heteronyms=None,
        encoding='utf-8',
        phoneme_probability: Optional[float] = None,
        mapping_file: Optional[str] = None,
        use_normalizer=True,
        use_zirneshan=True,
        use_hamnevise=True,
        ipa=True,
        stress=True
    ):
        """
        Persian G2P module using VaguyePipeline.
        """
        super().__init__(phoneme_dict=phoneme_dict, mapping_file=mapping_file)
        
        self.pipeline = VaguyePipeline(ipa=ipa, stress=stress)
        self.use_normalizer = use_normalizer
        self.use_zirneshan = use_zirneshan
        self.use_hamnevise = use_hamnevise
        
    def __call__(self, text):
        """
        Returns:
            phoneme_seq (str or list): The phonemized text.
        """
        # The pipeline handles normalization, hamnevise, zirneshan internally if initialized
        # We can control execution flow here if we want to bypass parts, but VaguyePipeline is structured to do all.
        # For now, we utilize the full pipeline as requested.
        
        result = self.pipeline(text, return_chunked=False)
        phoneme_seq = result["phonemes"]
        
        # Ensure compatibility with Tokenizer (extract string from result dict)
        return phoneme_seq
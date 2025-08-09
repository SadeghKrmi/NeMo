"""
Enhanced Persian Phoneme Tokenizer with emotion/style control and pause handling
"""
import re
import string
import logging
import itertools
import unicodedata
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional, Tuple, Dict

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo.collections.tts.g2p.models.fa_ir_persian.normalizer import PersianNormalizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

normalizer = PersianNormalizer()

def persian_text_preprocessing(text: str) -> str:
    """
    Preprocess Persian text with normalization.
    
    Args:
        text (str): the original input sentence.
    
    Returns:
        normalized text (str).
    """
    return normalizer.normalize(text)


# class BaseTokenizer(ABC):
#     PAD, BLANK, OOV = '<pad>', '<blank>', '<oov>'

#     def __init__(self, tokens, *, pad=PAD, blank=BLANK, oov=OOV, sep='', add_blank_at=None):
#         """Abstract class for creating an arbitrary tokenizer to convert string to list of int tokens.
#         Args:
#             tokens: List of tokens.
#             pad: Pad token as string.
#             blank: Blank token as string.
#             oov: OOV token as string.
#             sep: Separation token as string.
#             add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
#                 if None then no blank in labels.
#         """
#         super().__init__()

#         tokens = list(tokens)
#         self.pad, tokens = len(tokens), tokens + [pad]  # Padding

#         if add_blank_at is not None:
#             self.blank, tokens = len(tokens), tokens + [blank]  # Reserved for blank from asr-model
#         else:
#             self.blank = None

#         self.oov, tokens = len(tokens), tokens + [oov]  # Out Of Vocabulary

#         if add_blank_at == "last":
#             tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
#             self.oov, self.blank = self.blank, self.oov

#         self.tokens = tokens
#         self.sep = sep

#         self._util_ids = {self.pad, self.blank, self.oov}
#         self._token2id = {l: i for i, l in enumerate(tokens)}
#         self._id2token = tokens

#     def __call__(self, text: str) -> List[int]:
#         return self.encode(text)

#     @abstractmethod
#     def encode(self, text: str) -> List[int]:
#         """Turns str text into int tokens."""
#         pass

#     def decode(self, tokens: List[int]) -> str:
#         """Turns ints tokens into str text."""
#         return self.sep.join(self._id2token[t] for t in tokens if t not in self._util_ids)


class PersianPhonemesTokenizer(BaseTokenizer):
    # fmt: off
    
    # Persian punctuation with their pause/emotion effects
    PERSIAN_PUNCT_MAP = {
        '،': '<pause_short>',     # Persian comma - short pause
        '؛': '<pause_medium>',    # Persian semicolon - medium pause
        '.': '<pause_long>',      # Period - long pause
        '!': '<surprise>',        # Exclamation - surprise/emphasis
        '؟': '<question>',        # Persian question mark
        '?': '<question>',        # English question mark (fallback)
        '…': '<pause_long>',      # Ellipsis - long pause
        ':': '<pause_medium>',    # Colon - medium pause
        '-': '<pause_short>',     # Dash - short pause
        '‒': '<pause_short>',     # En dash - short pause
    }
    
    # Other punctuation marks (without special effects)
    OTHER_PUNCT = (
        '/', '"', '(', ')', '[', ']', 
        '{', '}', '«', '»', '"', '"',
        '‹', '›', ';', ','
    )
    
    # Emotion and style control tokens
    EMOTION_TOKENS = (
        '<happy>',
        '<sad>',
        '<angry>',
        '<whisper>',
        '<shout>',
        '<neutral>',
        '<excited>',
        '<calm>',
        '<fearful>',
        '<disgusted>',
    )
    
    # Pause control tokens
    PAUSE_TOKENS = (
        '<pause_short>',
        '<pause_medium>',
        '<pause_long>',
        '<pause_extra_long>',
    )
    
    # Additional control tokens from punctuation
    PUNCT_CONTROL_TOKENS = (
        '<question>',
        '<surprise>',
    )
    
    # Speed control tokens
    SPEED_TOKENS = (
        '<speed_slow>',
        '<speed_normal>',
        '<speed_fast>',
    )
    
    # Non-IPA phoneme system for Persian
    VOWELS = (
        "A", "Λ", "ą", "ę", "ó", "O", 
        "u", "E", "i", "I", "U", "a",
        "e", "o",
    )
    
    CONSONANTS = (
        "b", "p", "t", "c", "j", "C", 
        "H", "x", "d", "D", "r", "z", 
        "ʒ", "s", "S", "ć", "Ć", "T",
        "Z", "ʔ", "G", "f", "q", "k",
        "l", "m", "n", "v", "V", "h", 
        "Y", "y", "Ą", "Ę", "Ó", "ð",
        "ʊ", "ɔ", "θ", "X", "g"
    )
    
    # Valid characters in the phoneme system
    VALID_PHONEME_CHARS = set("bćtHEhqTĄ‌/mfkxĆUoZąCcay٬jlpnD?ʔAðręSdXvOIVsΛgYe'ɔGzuʊʒói")
    
    # fmt: on

    def __init__(
        self,
        g2p,
        punct=True,
        stresses=False,
        chars=False,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        oov=BaseTokenizer.OOV,
        sep='|',
        add_blank_at=None,
        pad_with_space=True,
        text_preprocessing_func=lambda text: persian_text_preprocessing(text),
        use_emotion_tokens=True,
        use_pause_tokens=True,
        use_speed_tokens=True,
    ):
        """Enhanced Persian phoneme-based tokenizer with emotion and control tokens.
        
        Args:
            g2p: Grapheme to phoneme module.
            punct: Whether to reserve grapheme for basic punctuation or not.
            stresses: Whether to use phonemes codes with stresses (0-2) or not.
            chars: Whether to additionally use chars together with phonemes.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order.
            pad_with_space: Whether to pad text with spaces at the beginning and end.
            text_preprocessing_func: Text preprocessing function.
            use_emotion_tokens: Whether to include emotion control tokens.
            use_pause_tokens: Whether to include pause control tokens.
            use_speed_tokens: Whether to include speed control tokens.
        """
        self.phoneme_probability = None
        if hasattr(g2p, "phoneme_probability"):
            self.phoneme_probability = g2p.phoneme_probability
        
        tokens = []
        self.space, tokens = len(tokens), tokens + [space]  # Space

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]  # Silence

        # Add consonants and vowels
        tokens.extend(self.CONSONANTS)
        vowels = list(self.VOWELS)

        if stresses:
            vowels = [f'{p}{s}' for p, s in itertools.product(vowels, (0, 1, 2))]
        tokens.extend(vowels)

        # Add valid phoneme system characters if needed
        if chars or self.phoneme_probability is not None:
            if not chars:
                logging.warning(
                    "phoneme_probability was not None, characters will be enabled even though "
                    "chars was set to False."
                )
            # Only add characters that are part of the Persian phoneme system
            tokens.extend(sorted(self.VALID_PHONEME_CHARS))
        
        if apostrophe:
            tokens.append("'")  # Apostrophe
        
        # Add control tokens
        if use_emotion_tokens:
            tokens.extend(self.EMOTION_TOKENS)
        
        if use_pause_tokens:
            tokens.extend(self.PAUSE_TOKENS)
            tokens.extend(self.PUNCT_CONTROL_TOKENS)  # Add punctuation-derived control tokens
            
        if use_speed_tokens:
            tokens.extend(self.SPEED_TOKENS)
        
        # Add punctuation
        if punct:
            # Add mapped punctuation tokens (already added via pause tokens)
            # Add other punctuation marks
            tokens.extend(self.OTHER_PUNCT)
        
        super().__init__(tokens, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.chars = chars if self.phoneme_probability is None else True
        self.punct = punct
        self.stresses = stresses
        self.pad_with_space = pad_with_space
        self.use_emotion_tokens = use_emotion_tokens
        self.use_pause_tokens = use_pause_tokens
        self.use_speed_tokens = use_speed_tokens

        self.text_preprocessing_func = text_preprocessing_func
        self.g2p = g2p

    def encode(self, text):
        """Encode text to token IDs."""
        text = self.text_preprocessing_func(text)
        
        # Pre-process to handle Persian punctuation
        if self.use_pause_tokens:
            text = self._apply_punct_mappings(text)
        
        # Extract control tokens and text segments
        segments, _ = self._extract_control_tokens(text)
        
        # Process each text segment through G2P
        processed_segments = []
        for segment in segments:
            if segment.startswith('<') and segment.endswith('>'):
                # This is a control token, keep as is
                processed_segments.append([segment])
            else:
                # This is text, process through G2P
                if segment.strip():  # Only process non-empty segments
                    g2p_result = self.g2p(segment)
                    processed_segments.append(g2p_result)
                elif segment == ' ':
                    # Preserve spaces between control tokens and text
                    processed_segments.append([' '])
        
        # Flatten the results
        g2p_text = []
        for segment in processed_segments:
            g2p_text.extend(segment)
        
        return self.encode_from_g2p(g2p_text, text)

    def _apply_punct_mappings(self, text: str) -> str:
        """Replace Persian punctuation with corresponding control tokens."""
        for punct, token in self.PERSIAN_PUNCT_MAP.items():
            text = text.replace(punct, f' {token} ')
        # Clean up multiple spaces
        text = ' '.join(text.split())
        return text

    def _extract_control_tokens(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract control tokens and text segments separately.
        
        Args:
            text: Input text possibly containing control tokens
            
        Returns:
            Tuple of (segments, control_tokens) where segments is a list of text/token segments
        """
        import re
        
        # Pattern to match control tokens
        control_pattern = r'(<[^>]+>)'
        
        # Split by control tokens, keeping the tokens
        parts = re.split(control_pattern, text)
        
        segments = []
        for part in parts:
            if part:  # Skip empty strings
                segments.append(part)
        
        return segments, []

    def encode_from_g2p(self, g2p_text: List[str], raw_text: Optional[str] = None):
        """
        Encodes text that has already been run through G2P.
        
        Args:
            g2p_text: G2P's output, mixture of phonemes and graphemes
            raw_text: original raw input
        """
        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)
        
        for p in g2p_text:
            # Handle control tokens first
            if p.startswith('<') and p.endswith('>'):
                if p in tokens:
                    ps.append(p)
                else:
                    # Log warning for unknown control token
                    logging.warning(f"Unknown control token: [{p}]. Skipping.")
                continue
            
            # Remove stress if not needed
            if p.isalnum() and len(p) == 3 and not self.stresses:
                p = p[:2]

            # Handle space
            if p == space:
                if len(ps) == 0 or ps[-1] != space:
                    ps.append(p)
            # Handle phonemes and valid characters
            elif (p.isalnum() or p == "'" or p in self.VALID_PHONEME_CHARS) and p in tokens:
                ps.append(p)
            # Handle other punctuation
            elif p in self.OTHER_PUNCT and self.punct and p in tokens:
                ps.append(p)
            # Skip unknown characters with warning
            elif p != space:
                message = f"Text: [{''.join(g2p_text)}] contains unknown char/phoneme: [{p}]."
                if raw_text is not None:
                    message += f" Original text: [{raw_text}]. Symbol will be skipped."
                logging.warning(message)
        
        # Remove trailing spaces
        while ps and ps[-1] == space:
            ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        return [self._token2id[p] for p in ps]
    
    def get_control_tokens(self) -> Dict[str, List[str]]:
        """Return available control tokens organized by category."""
        control_tokens = {}
        
        if self.use_emotion_tokens:
            control_tokens['emotions'] = list(self.EMOTION_TOKENS)
        
        if self.use_pause_tokens:
            control_tokens['pauses'] = list(self.PAUSE_TOKENS)
            control_tokens['punct_controls'] = list(self.PUNCT_CONTROL_TOKENS)
            control_tokens['punct_mappings'] = dict(self.PERSIAN_PUNCT_MAP)
        
        if self.use_speed_tokens:
            control_tokens['speeds'] = list(self.SPEED_TOKENS)
            
        return control_tokens
    
    def debug_tokens(self, text: str):
        """Debug method to show token processing steps."""
        print(f"Original text: {text}")
        
        # Check if control tokens are in vocabulary
        print("\nControl tokens in vocabulary:")
        for token in self.EMOTION_TOKENS + self.PAUSE_TOKENS + self.SPEED_TOKENS:
            if token in self._token2id:
                print(f"  {token}: ID {self._token2id[token]}")
            else:
                print(f"  {token}: NOT IN VOCABULARY!")
        
        # Show processing steps
        text = self.text_preprocessing_func(text)
        print(f"\nAfter preprocessing: {text}")
        
        if self.use_pause_tokens:
            text = self._apply_punct_mappings(text)
            print(f"After punct mapping: {text}")
        
        segments, _ = self._extract_control_tokens(text)
        print(f"\nSegments: {segments}")
        
        return segments
    
    @contextmanager
    def set_phone_prob(self, prob):
        """Context manager to temporarily set phoneme probability."""
        if hasattr(self.g2p, "phoneme_probability"):
            old_prob = self.g2p.phoneme_probability
            self.g2p.phoneme_probability = prob
        try:
            yield
        finally:
            if hasattr(self.g2p, "phoneme_probability"):
                self.g2p.phoneme_probability = old_prob
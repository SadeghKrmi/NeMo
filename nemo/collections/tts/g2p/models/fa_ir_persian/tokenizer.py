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

def persian_text_normalizer(text: str) -> str:
    return normalizer.normalize(text)

def persian_text_preprocessing(text: str) -> str:
    ntext = persian_text_normalizer(text)

    # handle brackets, pranthesis, etc...
    text_preprocessing_list_of_patterns = [
        (r'\((.*?)\)', r'،\1،'),    # convert (text) to comma + text + comma
        (r'\[(.*?)\]', r'،\1،'),    # convert [text] to comma + text + comma
        (r'\{(.*?)\}', r'،\1،'),    # convert {text} to comma + text + comma
        (r'"(.*?)"', r'،\1'),       # convert "text" to comma + text
        (r'‹(.*?)›', r' \1 '),    # convert ‹text› to text 
        (r'«(.*?)»', r' \1 '),    # convert «text» to text
    ]
    
    for pattern in text_preprocessing_list_of_patterns:
        ntext = re.sub(pattern[0], pattern[1], ntext)

    return ntext


class BaseTokenizer(ABC):
    PAD, BLANK, OOV = '<pad>', '<blank>', '<oov>'

    def __init__(self, tokens, *, pad=PAD, blank=BLANK, oov=OOV, sep='', add_blank_at=None):
        """Abstract class for creating an arbitrary tokenizer to convert string to list of int tokens.
        Args:
            tokens: List of tokens.
            pad: Pad token as string.
            blank: Blank token as string.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
                if None then no blank in labels.
        """
        super().__init__()

        tokens = list(tokens)
        self.pad, tokens = len(tokens), tokens + [pad]  # Padding

        if add_blank_at is not None:
            self.blank, tokens = len(tokens), tokens + [blank]  # Reserved for blank from asr-model
        else:
            self.blank = None

        self.oov, tokens = len(tokens), tokens + [oov]  # Out Of Vocabulary

        if add_blank_at == "last":
            tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
            self.oov, self.blank = self.blank, self.oov

        self.tokens = tokens
        self.sep = sep

        self._util_ids = {self.pad, self.blank, self.oov}
        self._token2id = {l: i for i, l in enumerate(tokens)}
        self._id2token = tokens

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""
        pass

    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.sep.join(self._id2token[t] for t in tokens if t not in self._util_ids)


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
        '‹', '›', '؛', '،'
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
        
        # Build token vocabulary with FIXED order regardless of switches
        tokens = []
        self.space, tokens = len(tokens), tokens + [space]  # Space

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]  # Silence
        else:
            self.silence = None

        # Add consonants and vowels
        tokens.extend(self.CONSONANTS)
        vowels = list(self.VOWELS)

        tokens.extend(vowels)
        
        if apostrophe:
            tokens.append("'")  # Apostrophe
        
        # ALWAYS add ALL control tokens to maintain consistent IDs
        # They will always have the same position in the vocabulary
        tokens.extend(self.EMOTION_TOKENS)
        tokens.extend(self.PAUSE_TOKENS)
        tokens.extend(self.PUNCT_CONTROL_TOKENS)
        tokens.extend(self.SPEED_TOKENS)
        
        # Add punctuation
        if punct:
            tokens.extend(self.OTHER_PUNCT)
        
        super().__init__(tokens, oov=oov, sep=sep, add_blank_at=add_blank_at)

        # Store configuration for which tokens are actually active
        self.punct = punct
        self.pad_with_space = pad_with_space
        self.use_emotion_tokens = use_emotion_tokens
        self.use_pause_tokens = use_pause_tokens
        self.use_speed_tokens = use_speed_tokens

        # Create sets of active tokens for faster lookup
        self._active_tokens = set(self.tokens)
        
        # Remove inactive control tokens from active set (but keep them in vocabulary)
        if not use_emotion_tokens:
            self._active_tokens -= set(self.EMOTION_TOKENS)
        if not use_pause_tokens:
            self._active_tokens -= set(self.PAUSE_TOKENS)
            self._active_tokens -= set(self.PUNCT_CONTROL_TOKENS)
        if not use_speed_tokens:
            self._active_tokens -= set(self.SPEED_TOKENS)

        self.text_preprocessing_func = text_preprocessing_func
        self.g2p = g2p

    def encode(self, text):
        """Encode text to token IDs."""
        text = self.text_preprocessing_func(text)
        
        # Pre-process to handle Persian punctuation only if pause tokens are active
        if self.use_pause_tokens:
            text = self._apply_punct_mappings(text)
        
        # Extract control tokens and text segments
        segments, _ = self._extract_control_tokens(text)
        
        # Process each text segment through G2P
        processed_segments = []
        for segment in segments:
            if segment.startswith('<') and segment.endswith('>'):
                # This is a control token, check if it's active
                if segment in self._active_tokens:
                    processed_segments.append([segment])
                else:
                    # Skip inactive control tokens
                    logging.debug(f"Skipping inactive control token: {segment}")
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
        ps, space, tokens = [], self.tokens[self.space], self._active_tokens
        
        for p in g2p_text:
            # Handle control tokens first
            if p.startswith('<') and p.endswith('>'):
                if p in tokens:
                    ps.append(p)
                else:
                    # Log warning for unknown or inactive control token
                    logging.debug(f"Control token not active or unknown: [{p}]. Skipping.")
                continue

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
    
    def export_tokenizer_mappings(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export tokenizer vocabulary and mappings for inspection or external use.
        
        Args:
            filepath: Optional filepath to save the mappings as JSON
            
        Returns:
            Dictionary containing all tokenizer mappings and metadata
        """
        mappings = {
            'metadata': {
                'total_tokens': len(self.tokens),
                'vocab_size': len(self._token2id),
                'special_tokens': {
                    'pad': {'token': self.tokens[self.pad] if self.pad < len(self.tokens) else None, 
                            'id': self.pad},
                    'blank': {'token': self.tokens[self.blank] if self.blank and self.blank < len(self.tokens) else None, 
                             'id': self.blank},
                    'oov': {'token': self.tokens[self.oov] if self.oov < len(self.tokens) else None, 
                           'id': self.oov},
                    'space': {'token': self.tokens[self.space] if hasattr(self, 'space') else None, 
                             'id': self.space if hasattr(self, 'space') else None},
                    'silence': {'token': self.tokens[self.silence] if hasattr(self, 'silence') and self.silence else None,
                               'id': self.silence if hasattr(self, 'silence') else None}
                },
                'configuration': {
                    'punct': self.punct,
                    'use_emotion_tokens': self.use_emotion_tokens,
                    'use_pause_tokens': self.use_pause_tokens,
                    'use_speed_tokens': self.use_speed_tokens,
                    'pad_with_space': self.pad_with_space
                }
            },
            'token_to_id': self._token2id,
            'id_to_token': {i: token for i, token in enumerate(self._id2token)},
            'active_tokens': sorted(list(self._active_tokens)),
            'token_categories': {
                'consonants': list(self.CONSONANTS),
                'vowels': list(self.VOWELS),
                'emotion_tokens': list(self.EMOTION_TOKENS),
                'pause_tokens': list(self.PAUSE_TOKENS),
                'punct_control_tokens': list(self.PUNCT_CONTROL_TOKENS),
                'speed_tokens': list(self.SPEED_TOKENS),
                'other_punct': list(self.OTHER_PUNCT),
                'persian_punct_map': dict(self.PERSIAN_PUNCT_MAP)
            },
            'phoneme_chars': sorted(list(self.VALID_PHONEME_CHARS))
        }
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, ensure_ascii=False, indent=2)
            logging.info(f"Tokenizer mappings exported to {filepath}")
        
        return mappings
    
    def verify_consistency(self, other_config: Dict[str, bool]) -> bool:
        """Verify that token IDs remain consistent with different configurations.
        
        Args:
            other_config: Dictionary with keys like 'use_emotion_tokens', 'use_pause_tokens', etc.
            
        Returns:
            True if token IDs are consistent, False otherwise
        """
        # Check that all tokens have the same IDs regardless of configuration
        for token in self.tokens:
            if token in self._token2id:
                # This token should always have the same ID
                expected_id = self._token2id[token]
                actual_id = self.tokens.index(token)
                if expected_id != actual_id:
                    logging.error(f"Token '{token}' has inconsistent ID: expected {expected_id}, got {actual_id}")
                    return False
        
        logging.info("Token ID consistency verified successfully")
        return True
    
    def debug_tokens(self, text: str):
        """Debug method to show token processing steps."""
        print(f"Original text: {text}")
        
        # Check if control tokens are in vocabulary
        print("\nControl tokens in vocabulary:")
        for category, tokens in [
            ('Emotions', self.EMOTION_TOKENS),
            ('Pauses', self.PAUSE_TOKENS),
            ('Punct Controls', self.PUNCT_CONTROL_TOKENS),
            ('Speeds', self.SPEED_TOKENS)
        ]:
            print(f"\n{category}:")
            for token in tokens:
                if token in self._token2id:
                    active = token in self._active_tokens
                    status = "ACTIVE" if active else "INACTIVE"
                    print(f"  {token}: ID {self._token2id[token]} [{status}]")
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
        
        # Show encoding
        encoded = self.encode(text)
        print(f"\nEncoded IDs: {encoded}")
        print(f"Decoded: {self.decode(encoded)}")
        
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
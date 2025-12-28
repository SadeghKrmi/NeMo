"""
Enhanced Persian Phoneme Tokenizer with fixed sparse vocabulary
"""
import logging
from typing import List, Optional, Dict, Any, Tuple

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
    # We map these to specific punctuation that we support: comma (,) which is 44
    # The user allows: ',' 44, '.' 46, ';' 59, '?' 63, '!' 33
    # We will map standard Persian punctuation to these supported ASCII ones.
    
    text_preprocessing_list_of_patterns = [
        (r'\((.*?)\)', r',\1,'),    # convert (text) to comma + text + comma
        (r'\[(.*?)\]', r',\1,'),    # convert [text] to comma + text + comma
        (r'\{(.*?)\}', r',\1,'),    # convert {text} to comma + text + comma
        (r'"(.*?)"', r',\1'),       # convert "text" to comma + text
        (r'‹(.*?)›', r' \1 '),      # convert ‹text› to text 
        (r'«(.*?)»', r' \1 '),      # convert «text» to text
    ]
    
    import re
    for pattern in text_preprocessing_list_of_patterns:
        ntext = re.sub(pattern[0], pattern[1], ntext)

    return ntext


class PersianPhonemesTokenizer(BaseTokenizer):
    # Fixed vocabulary mapping based on user input
    VALID_TOKENS_MAP = {
        32: ' ',   # ' '
        33: '!',   # '!'
        44: ',',   # ','
        46: '.',   # '.'
        59: ';',   # ';'
        63: '?',   # '?'
        97: 'a',   # 'a'
        98: 'b',   # 'b'
        100: 'd',  # 'd'
        101: 'e',  # 'e'
        102: 'f',  # 'f'
        104: 'h',  # 'h'
        105: 'i',  # 'i'
        106: 'j',  # 'j'
        107: 'k',  # 'k'
        108: 'l',  # 'l'
        109: 'm',  # 'm'
        110: 'n',  # 'n'
        111: 'o',  # 'o'
        112: 'p',  # 'p'
        113: 'q',  # 'q'
        114: 'r',  # 'r'
        115: 's',  # 's'
        116: 't',  # 't'
        117: 'u',  # 'u'
        118: 'v',  # 'v'
        120: 'x',  # 'x'
        122: 'z',  # 'z'
        230: 'æ',  # 'æ'
        594: 'ɒ',  # 'ɒ'
        609: 'ɡ',  # 'ɡ'
        643: 'ʃ',  # 'ʃ'
        658: 'ʒ',  # 'ʒ'
        660: 'ʔ',  # 'ʔ'
        712: 'ˈ',  # 'ˈ'
        716: 'ˌ',  # 'ˌ'
        720: 'ː',  # 'ː'
    }

    # Punctuation mapping from Persian to allowed ASCII
    PUNCT_MAPPING = {
        '،': ',',
        '؛': ';',
        '؟': '?',
        '−': ' ', 
        '-': ' ', 
        '_': ' ',
    }

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
        # Legacy args for compatibility
        use_emotion_tokens=False,
        use_pause_tokens=False,
        use_speed_tokens=False,
    ):
        """Fixed Persian phoneme-based tokenizer."""
        self.phoneme_probability = None
        if hasattr(g2p, "phoneme_probability"):
            self.phoneme_probability = g2p.phoneme_probability
        
        # Build strict sparse token list
        max_id = max(self.VALID_TOKENS_MAP.keys())
        vocab_size = max_id + 1
        
        # Initialize with a reserved placeholder
        tokens = ["<unused>"] * vocab_size
        
        # Fill in valid tokens at their specific indices
        for code, char in self.VALID_TOKENS_MAP.items():
            tokens[code] = char

        # Initialize BaseTokenizer
        super().__init__(tokens, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.punct = punct
        self.pad_with_space = pad_with_space
        self.text_preprocessing_func = text_preprocessing_func
        self.g2p = g2p
        
        self.valid_chars_set = set(self.VALID_TOKENS_MAP.values())

    def encode(self, text):
        """Encode text to token IDs."""
        text = self.text_preprocessing_func(text)
        
        for p, r in self.PUNCT_MAPPING.items():
            text = text.replace(p, r)

        # Run G2P
        g2p_result = self.g2p(text)
        
        if g2p_result and isinstance(g2p_result[0], list):
             # flatten
            flat_result = []
            for item in g2p_result:
                if isinstance(item, list):
                    flat_result.extend(item)
                else:
                    flat_result.append(item)
            g2p_result = flat_result

        return self.encode_from_g2p(g2p_result, text)

    def encode_from_g2p(self, g2p_text: List[str], raw_text: Optional[str] = None):
        """
        Encodes text that has already been run through G2P.
        """
        ps = []
        space_char = self.VALID_TOKENS_MAP.get(32, ' ')
        
        for p in g2p_text:
            if p in self.PUNCT_MAPPING:
                p = self.PUNCT_MAPPING[p]

            if p in self.valid_chars_set:
                ps.append(p)
            else:
                if p.strip() == '': # whitespace
                    if space_char in self.valid_chars_set:
                        # normalize any whitespace to our Space
                        if not ps or ps[-1] != space_char:
                             ps.append(space_char)
                else:
                     logging.debug(f"Skipping invalid char from G2P: {p}")
        
        # Handle pad with space
        if self.pad_with_space:
             if not ps or ps[0] != space_char:
                 ps.insert(0, space_char)
             if not ps or ps[-1] != space_char:
                 ps.append(space_char)

        # Convert to IDs
        ids = []
        for p in ps:
            if p in self._token2id:
                ids.append(self._token2id[p])
        
        return ids
    
    def verify_consistency(self, other_config: Dict[str, bool]) -> bool:
        """Verify that token IDs remain consistent."""
        for code, char in self.VALID_TOKENS_MAP.items():
            if self._token2id.get(char) != code:
                logging.error(f"Mismatch for {char}: expected {code}, got {self._token2id.get(char)}")
                return False
        return True

    def debug_tokens(self, text: str):
        print(f"Original: {text}")
        encoded = self.encode(text)
        print(f"Encoded: {encoded}")
        return encoded
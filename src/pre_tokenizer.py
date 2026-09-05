from collections import Counter
from typing import Dict, List, Tuple
import regex as re

from .vocabulary import Vocabulary

pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class PreTokenizer:
    def __init__(self):
        self.vocabulary = Vocabulary()

    def pre_tokenize(self, text: str) -> Dict[Tuple[int, ...], int]:
        """Split text and count frequencies of each byte tuple."""
        slices = self.split(text)
        return dict(
            Counter(tuple(self.vocabulary.encode(s)) for s in slices))

    def split(self, text: str) -> List[str]:
        return re.findall(pattern, text)
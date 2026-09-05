from typing import Dict, Tuple, Optional
from collections import defaultdict
from itertools import pairwise

from .pre_tokenizer import PreTokenizer
from .vocabulary import Vocabulary


class Tokenizer:
    def __init__(self, vocabulary_size: int):
        self.pre_tokenize = PreTokenizer()
        self.vocab = Vocabulary()

        self.vocabulary_size = vocabulary_size - 256
        self.merges = dict()

    def step(self):
        winner = self._get_winner_pair()
        self.merges.add(winner)
        new_token = vocabulary.add_token(best_key)
        slef._rewriting(new_token)

    def _get_winner_pair(self) -> Optional[Tuple[int, int]]:
        pairs = defaultdict(int)

        # 1. Accumulate pair frequencies
        for t, count in tokens.items():
            # pairwise(t) generates consecutive (t[i], t[i+1]) in C
            for pair in pairwise(t):
                pairs[pair] += count

        if not pairs:
            return None

        # 2. Extract max value and tie-break in pure C
        max_val = max(pairs.values())
        return min(k for k, v in pairs.items() if v == max_val)

    def _rewriting(self, new_token: int):
        for token in self.tokens:
            i = 0
            token = list(token)
            while i < len(token):
                if (token[i], token[i+1]) == best_pair:
                    token[i:i+1] = new_token
                i += 1


    def train(self, text_data, target_vocab_size):
        self.tokens = ptoken.pre_tokenize(text)
        i = 0
        while i < self.vocabulary_size:
            self.step()

    def encode(self, text):
        pass

    def decode(self, ids):
        pass

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass

from typing import Dict, Tuple, Optional, Any, List, Set

import json
from collections import defaultdict
from itertools import pairwise
import regex as re  # type: ignore [import-untyped]

from .pre_tokenizer import PreTokenizer
from .vocabulary import Vocabulary


class Tokenizer:
    def __init__(self, vocabulary_size: int = 50000):
        self.pre_tokenize = PreTokenizer()
        self.vocab = Vocabulary()

        self.vocabulary_size = vocabulary_size
        self.num_merges = self.vocabulary_size - 256
        self.metadata = {
            "version": "0.1.0",
            "model_type": "byte-bpe",
            "vocab_size": self.vocabulary_size
            }
        self.special_tokens: Dict[str, int] = {}

        self.merges: Dict[Tuple[int, int], int] = dict()
        self.chunks: Dict[Tuple[int, ...], int] = dict()

    def _step(self) -> bool:
        winner = self._get_winner_pair()
        if winner is None:
            return False
        new_token = self.vocab.add_token(winner)
        self.merges[winner] = new_token
        self._rewriting(winner, new_token)
        return True

    def _get_winner_pair(self) -> Optional[Tuple[int, int]]:
        pairs: Dict[Tuple[int, int], int] = defaultdict(int)

        # 1. Accumulate pair frequencies
        for c, count in self.chunks.items():
            # pairwise(c) generates consecutive (c[i], c[i+1]) in C
            for pair in pairwise(c):
                pairs[pair] += count

        if not pairs:
            return None

        # 2. Extract max value and tie-break in pure C
        max_val = max(pairs.values())
        if max_val < 2:
            return None
        return min(k for k, v in pairs.items() if v == max_val)

    def _rewriting(self,
                   best_pair: Tuple[int, int],
                   new_token: int) -> None:
        new_chunks = {}
        for chunk, count in self.chunks.items():
            new_chunk = []
            i = 0
            while i < len(chunk):
                if (i < len(chunk) - 1 and
                        (chunk[i], chunk[i + 1]) == best_pair):
                    new_chunk.append(new_token)
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            new_chunks[tuple(new_chunk)] = count
        self.chunks = new_chunks

    def _generate_json_vocab(self) -> Dict[int, List[int]]:
        return {
            k: list(v)
            for k, v in self.vocab.id_to_bytes.items()
            }

    def _generate_merges(self):
        return sorted(self.merges, key=self.merges.get)

    def _generate_json_tokenizer(self) -> Dict[str, Any]:
        tokenizer_json: Dict[str, Any] = {}
        tokenizer_json["metadata"] = self.metadata
        tokenizer_json["vocabulary"] = self._generate_json_vocab()
        tokenizer_json["merges"] = self._generate_merges()
        tokenizer_json["special_tokens"] = self.special_tokens
        return tokenizer_json

    def _load_vocab_from_json_tokenizer(self, vocab: Dict[int, List[int]]):
        self.vocab.id_to_bytes = {int(k): bytes(v) for k, v in vocab.items()}
        self.vocab.bytes_to_id = {bytes(v): int(k) for k, v in vocab.items()}

    def _load_merges(self, merges: List[List[int]]):
        self.merges = {(m[0], m[1]): 256 + i for i, m in enumerate(merges)}

    def _rewrite_slice(self, merge, raw_text: List[int]) -> List[int]:
        new_slice = []
        i = 0
        while i < len(raw_text):
            if (i < len(raw_text) - 1 and
                    (raw_text[i], raw_text[i + 1]) == merge):
                new_slice.append(self.merges[merge])
                i += 2
            else:
                new_slice.append(raw_text[i])
                i += 1
        return new_slice

    def _split_special_tokens(self,
                              text: str,
                              allowed_special: Set[str] | str) -> List[str]:
        if not self.special_tokens:
            return [text]

        if isinstance(allowed_special, set):
            allowed = {st for st in allowed_special
                       if st in self.special_tokens}
        elif allowed_special == "all":
            allowed = set(self.special_tokens.keys())
        else:
            raise ValueError("Invalid value for allowed_special")

        if not allowed:
            return [text]
        pattern = f"({'|'.join(re.escape(st) for st in allowed)})"
        return re.split(pattern, text)

    def train(self,
              text_data: str,
              target_vocab_size: Optional[int] = None) -> None:
        if target_vocab_size is not None:
            self.vocabulary_size = target_vocab_size
            self.num_merges = self.vocabulary_size - 256
        self.chunks = self.pre_tokenize.pre_tokenize(text_data)
        for _ in range(self.num_merges):
            if not self._step():
                break
        self.save("tokenizer.json")

    def encode(self,
               text: str,
               allowed_special: Set[str] | str = "all"
               ) -> List[int]:
        list_tokens = []
        split_chunks = self._split_special_tokens(text, allowed_special)
        for chunk in split_chunks:
            if chunk in self.special_tokens:
                list_tokens.extend([self.special_tokens[chunk]])
                continue

            pretokenized_slices = self.pre_tokenize.split(chunk)
            for slice_ in pretokenized_slices:
                raw_text = self.vocab.encode(slice_)

                while len(raw_text) >= 2:
                    # Find pairs present in the current slice
                    pairs = [
                        (raw_text[i], raw_text[i+1])
                        for i in range(len(raw_text) - 1)
                        ]
                    # Find which pair in this slice has the lowest
                    # rank in self.merges
                    candidate_pairs = [p for p in pairs if p in self.merges]
                    if not candidate_pairs:
                        break
                    # Highest priority = earliest learned
                    best_pair = min(candidate_pairs,
                                    key=lambda p: self.merges[p])
                    raw_text = self._rewrite_slice(best_pair, raw_text)
                list_tokens.extend(raw_text)

        return list_tokens

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            ids = [
                id_ for id_ in ids
                if id_ not in self.special_tokens.values()
                ]
        return self.vocab.decode(ids)

    def register_special_token(self,
                               token_str: str,
                               token_id: Optional[int] = None):
        if token_id is None:
            token_id = len(self.vocab.id_to_bytes)
        self.special_tokens[token_str] = token_id
        token_bytes = token_str.encode("utf-8")
        self.vocab.id_to_bytes[token_id] = token_bytes
        self.vocab.bytes_to_id[token_bytes] = token_id

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self._generate_json_tokenizer(), f, indent=4)

    def load(self, file_path: str):
        with open(file_path, "r") as f:
            tokenizer_json = json.load(f)
        self.special_tokens = tokenizer_json["special_tokens"]
        self.metadata = tokenizer_json["metadata"]
        self._load_vocab_from_json_tokenizer(tokenizer_json["vocabulary"])
        self._load_merges(tokenizer_json["merges"])

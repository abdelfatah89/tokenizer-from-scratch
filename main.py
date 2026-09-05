import numpy as np

from src.pre_tokenizer import PreTokenizer
from src.vocabulary import Vocabulary


def main():
    ptoken = PreTokenizer()
    vocabulary = Vocabulary()

    vocab_id_to_byte = vocabulary.id_to_bytes
    vocab_byte_to_id = vocabulary.bytes_to_id

    text = """The quick brown fox jumps over the lazy dog. The dog wasn't really that lazy, but the fox was very, very quick!
In 2024, the fox said, "I'll do it again." And indeed, the fox jumped 10 times, then 20 times, then 30 times.

Why did the fox jump?
    1. Because the fox could.
    2. Because it's what foxes do.
    3. Because the lazy dog didn't care at all.

The end. It's truly the end! Or is it? We'll see... we'll definitely see."""

    tokens = ptoken.pre_tokenize(text)
    pairs = dict()
    for t in tokens:
        token_len = len(t)
        if token_len == 1:
            continue
        for i in range(token_len):
            if i + 1 < token_len:
                pair = (t[i], t[i+1])
                pairs[pair] = tokens[t]

    if not pairs:
        return
    max_val = max(pairs.values())
    best_key = min(k for k, v in pairs.items() if v == max_val)
    vocabulary.add_token(best_key)
    print(vocabulary.encode("\n "))



if __name__ == "__main__":
    main()

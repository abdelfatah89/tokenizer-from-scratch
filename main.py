from src.pre_tokenizer import PreTokenizer
from src.bytes_mapper import BytesMapper


def main():
    ptoken = PreTokenizer()
    maper = BytesMapper()
    text = """The quick brown fox jumps over the lazy dog. The dog wasn't really that lazy, but the fox was very, very quick!
In 2024, the fox said, "I'll do it again." And indeed, the fox jumped 10 times, then 20 times, then 30 times.

Why did the fox jump?
    1. Because the fox could.
    2. Because it's what foxes do.
    3. Because the lazy dog didn't care at all.

The end. It's truly the end! Or is it? We'll see... we'll definitely see."""

    tokens = ptoken.pre_tokenize(text)
    for t in tokens:
        print(maper.decode(list(t)), ":", tokens[t])


if __name__ == "__main__":
    main()

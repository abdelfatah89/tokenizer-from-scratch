import os
import sys
from src.tokenizer import Tokenizer


def run_unit_tests(tok: Tokenizer):
    print("=" * 60)
    print("🧪 RUNNING VERIFICATION & EDGE-CASE TEST SUITE")
    print("=" * 60)

    test_cases = [
        "Hello world!",
        "Don't hesitate; it's 2026.",
        "   Leading, trailing, and    multiple spaces.   \n\n\t",
        "Multi-byte UTF-8 test: こんにちは! Bonjour! مرحبًا! 🚀🌟",
        "Numbers and symbols: #42 costs $99.99 (100% test).",
        "",  # Empty string
        "a" * 30,  # Repetition
    ]

    passed = 0
    for i, test in enumerate(test_cases, start=1):
        encoded = tok.encode(test)
        decoded = tok.decode(encoded)

        # Invertibility assertion
        if decoded == test:
            print(f"  [PASS] Test {i}: {repr(test[:35])}...")
            passed += 1
        else:
            print(f"  [FAIL] Test {i}: {repr(test)}")
            print(f"         Expected: {repr(test)}")
            print(f"         Got:      {repr(decoded)}")

    print("-" * 60)
    print(f"Summary: {passed}/{len(test_cases)} tests passed.")
    assert passed == len(test_cases), "One or more tests failed!"
    print("All invertibility checks passed successfully!\n")


def interactive_cli(tok: Tokenizer):
    print("=" * 60)
    print("✨ INTERACTIVE TOKENIZER SHOWCASE / CLI")
    print("Type any text to tokenize it.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nEnter text > ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        token_ids = tok.encode(user_input)
        reconstructed = tok.decode(token_ids)

        raw_bytes = len(user_input.encode("utf-8"))
        num_tokens = len(token_ids)
        compression = raw_bytes / num_tokens if num_tokens > 0 else 0.0

        print("\n--- Results ---")
        print(f"Raw Bytes Length : {raw_bytes} bytes")
        print(f"Token IDs Length : {num_tokens} tokens")
        print(f"Compression Ratio: {compression:.2f} bytes/token")
        print(f"Token IDs        : {token_ids}")
        print(f"Reconstructed    : {reconstructed}")
        print(f"Match Verified   : {'✅ Yes' if reconstructed == user_input else '❌ No'}")


def main():
    save_file = "tokenizer.json"
    tok = Tokenizer()

    if os.path.exists(save_file):
        print(f"Loading existing model from '{save_file}'...")
        tok.load(save_file)
    else:
        print("No saved tokenizer found. Training a baseline model...")
        corpus = """
        Byte-Pair Encoding (BPE) is a subword tokenization algorithm.
        It starts with a base vocabulary of 256 raw bytes and repeatedly merges
        the most frequent adjacent pairs. In modern large language models,
        tokenization bridges continuous human text and discrete neural representations.
        Don't skip boundaries! Numbers like 2026 and contractions like it's, you'll,
        and they're must be handled carefully.
        Hello, world! Keep learning and building from scratch.
        """
        tok.train(corpus, target_vocab_size=350)
        print("Training complete and saved to disk.")

    # 1. Run Automated Tests
    run_unit_tests(tok)

    # 2. Launch Interactive Showcase
    interactive_cli(tok)


if __name__ == "__main__":
    main()

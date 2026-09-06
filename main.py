import os
from src.tokenizer import Tokenizer


def run_unit_tests(tok: Tokenizer):
    print("=" * 65)
    print("🧪 1. RUNNING STANDARD & EDGE-CASE TEST SUITE")
    print("=" * 65)

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

        if decoded == test:
            print(f"  [PASS] Test {i:02d}: {repr(test[:32])}...")
            passed += 1
        else:
            print(f"  [FAIL] Test {i:02d}: {repr(test)}")
            print(f"         Expected: {repr(test)}")
            print(f"         Got:      {repr(decoded)}")

    print("-" * 65)
    print(f"Standard tests: {passed}/{len(test_cases)} passed.\n")
    assert passed == len(test_cases), "Standard edge-case tests failed!"

    # ==========================================
    # 2. SPECIAL TOKENS TEST SUITE
    # ==========================================
    print("=" * 65)
    print("🛡️  2. RUNNING SPECIAL TOKENS TEST SUITE")
    print("=" * 65)

    # Ensure test special tokens are registered
    if "<|endoftext|>" not in tok.special_tokens:
        tok.register_special_token("<|endoftext|>")
    if "<pad>" not in tok.special_tokens:
        tok.register_special_token("<pad>")

    eot_id = tok.special_tokens["<|endoftext|>"]
    pad_id = tok.special_tokens["<pad>"]

    special_test = "Hello world! <|endoftext|> Let's build. <pad>"
    encoded_specials = tok.encode(special_test, allowed_special="all")

    print(f"  Input sentence : {repr(special_test)}")
    print(f"  Encoded IDs    : {encoded_specials}")

    # Check 1: Ensure special IDs are present atomically
    assert eot_id in encoded_specials, (
        "Error: <|endoftext|> was shredded instead of preserved!"
    )
    assert pad_id in encoded_specials, (
        "Error: <pad> was shredded instead of preserved!"
    )
    print("  [PASS] Atomicity Check: Special tokens retained "
          "exact assigned IDs.")

    # Check 2: Full decode preserves special tokens
    decoded_full = tok.decode(encoded_specials, skip_special_tokens=False)
    assert decoded_full == special_test, (
        f"Mismatch: expected '{special_test}', got '{decoded_full}'"
    )
    print("  [PASS] Full Decode: Decoded text matches original "
          "with special tokens.")

    # Check 3: Stripped decode skips special tokens cleanly
    decoded_stripped = tok.decode(encoded_specials, skip_special_tokens=True)
    assert "<|endoftext|>" not in decoded_stripped, (
        "Error: <|endoftext|> was not skipped!"
    )
    assert "<pad>" not in decoded_stripped, "Error: <pad> was not skipped!"
    print(f"  [PASS] Stripped Decode: Successfully stripped specials -> "
          f"{repr(decoded_stripped)}")

    print("-" * 65)
    print("All special token assertions passed successfully!\n")


def interactive_cli(tok: Tokenizer):
    print("=" * 65)
    print("✨ INTERACTIVE TOKENIZER SHOWCASE / CLI")
    print("=" * 65)
    print("Commands:")
    print("  - Type any text to tokenize and inspect it.")
    print("  - ':add <token>' to register a new special token.")
    print("  - ':list'        to show all registered special tokens.")
    print("  - 'exit'/'quit'  to terminate.")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nEnter text > ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        cleaned = user_input.strip()
        if cleaned.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Command: List Special Tokens
        if cleaned == ":list":
            print("\nRegistered Special Tokens:")
            for token, t_id in tok.special_tokens.items():
                print(f"  {token:<20} -> ID: {t_id}")
            continue

        # Command: Add Special Token
        if cleaned.startswith(":add "):
            new_st = cleaned[5:].strip()
            if not new_st:
                print("Error: Empty token string.")
                continue
            tok.register_special_token(new_st)
            print(f"Registered special token '{new_st}' with ID: "
                  f"{tok.special_tokens[new_st]}")
            continue

        if not user_input:
            continue

        # Tokenization
        token_ids = tok.encode(user_input, allowed_special="all")
        reconstructed_full = tok.decode(token_ids, skip_special_tokens=False)
        reconstructed_stripped = tok.decode(
            token_ids, skip_special_tokens=True)

        raw_bytes = len(user_input.encode("utf-8"))
        num_tokens = len(token_ids)
        compression = raw_bytes / num_tokens if num_tokens > 0 else 0.0

        # Check for special tokens in this input
        specials_found = [
            st for st, s_id in tok.special_tokens.items() if s_id in token_ids
        ]

        print("\n--- Results ---")
        print(f"Raw Bytes Length   : {raw_bytes} bytes")
        print(f"Token IDs Length   : {num_tokens} tokens")
        print(f"Compression Ratio  : {compression:.2f} bytes/token")
        print(f"Token IDs          : {token_ids}")
        print(f"Decoded (Full)     : {reconstructed_full}")

        if specials_found:
            print(f"Detected Specials  : {specials_found}")
            print(f"Decoded (Stripped) : {reconstructed_stripped}")

        print(f"Match Verified     : "
              f"{'✅ Yes' if reconstructed_full == user_input else '❌ No'}")


def main():
    save_file = "tokenizer.json"
    tok = Tokenizer()

    if os.path.exists(save_file):
        print(f"Loading existing model from '{save_file}'...")
        tok.load(save_file)
    else:
        print("No saved tokenizer found. Training baseline model...")
        corpus = """
        Byte-Pair Encoding (BPE) is a subword tokenization algorithm.
        It starts with a base vocabulary of 256 raw bytes and repeatedly merges
        the most frequent adjacent pairs. In modern large language models,
        tokenization bridges continuous human text and discrete
        neural representations.
        Don't skip boundaries! Numbers like 2026 and contractions like it's,
        you'll, and they're must be handled carefully.
        Hello, world! Keep learning and building from scratch.
        """
        tok.train(corpus, target_vocab_size=350)
        tok.register_special_token("<|endoftext|>")
        tok.register_special_token("<pad>")
        tok.save(save_file)
        print("Training complete and saved to disk.")

    # 1. Run Automated Tests
    run_unit_tests(tok)

    # 2. Launch Interactive Showcase
    interactive_cli(tok)


if __name__ == "__main__":
    main()

# 🪙 tokenizer-from-scratch

A clean, standalone, zero-dependency implementation of a **Byte-Level Byte-Pair Encoding (BPE)** Tokenizer built completely from first principles in Python.

Inspired by the architectures of modern LLM tokenizers (GPT-4, LLaMA, Mistral), this project demonstrates how raw bytes are transformed into bounded chunks, statistically merged into subwords, and serialized into a custom `tokenizer.json`.

---

## ✨ Features

- **Pure Byte-Level Encoding**: Base vocabulary of 256 bytes (`0x00` - `0xFF`). Mathematically zero Out-Of-Vocabulary (`<unk>`) errors.
- **Regex Boundary Firewalls**: Isolates punctuation, contractions, whitespace, and numerical chunks to prevent nonsensical subword merges.
- **Deterministic Tie-Breaking**: Produces identical vocabulary IDs and merge priority ranks across any machine.
- **Lossless Invertibility**: Guarantees that `decode(encode(text)) == text` across multi-byte UTF-8 scripts, symbols, and emojis.
- **Standalone JSON Serialization**: Exports unified `tokenizer.json` containing metadata, merges, and vocabulary tables.
- **Zero Heavy ML Dependencies**: Built strictly using standard Python primitives and lightweight regular expressions.

---

## 🏗️ Architecture & Pipeline

```
Raw Text
   │
   ▼
[ PreTokenizer ]    --> Regex-based chunking ('s, words, numbers, punctuation)
   │
   ▼
[ Byte Conversion ] --> Converts chunks into raw UTF-8 byte tuples (0–255)
   │
   ▼
[ BPE Training ]    --> Iteratively pairs, tie-breaks, and assigns subword IDs
   │
   ▼
[ Serializer ]      --> Exports state into `tokenizer.json`
   │
   ▼
[ Inference Engine] --> Priority-based subword replacement (Encode & Decode)
```

---

## 📁 Project Structure

```
tokenizer-from-scratch/
├── vocabulary.py       # Base 256-byte table and UTF-8 byte stream handlers
├── pre_tokenizer.py    # Regex chunking firewalls and frequency aggregator
├── tokenizer.py        # Core BPE training loop, merge rules, and JSON serialization
├── main.py             # Verification tests and Interactive CLI showcase
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the regex engine:

```bash
git clone https://github.com/your-username/tokenizer-from-scratch.git
cd tokenizer-from-scratch
pip install regex
```

### 2. Basic Usage

```python
from tokenizer import Tokenizer

# 1. Initialize
tok = Tokenizer()

# 2. Train on raw text
training_data = """
Byte-Pair Encoding is a subword tokenization algorithm.
It merges the most frequent pairs of bytes iteratively.
"""
tok.train(training_data, target_vocab_size=300)

# 3. Encode & Decode
text = "Byte-Pair tokenization!"
encoded = tok.encode(text)
decoded = tok.decode(encoded)

print(f"Tokens: {encoded}")
print(f"Decoded: '{decoded}'")
assert decoded == text
```

---

## 🧪 Testing & Interactive CLI

To run the automated validation suite and start the interactive CLI:

```bash
python main.py
```

---

## 📊 How It Works

1. **Phase 1 (Base Alphabet)**: Seeds tokens `0..255` mapped to raw byte literals.
2. **Phase 2 (Pre-tokenization)**: Chunks text using GPT-style regular expressions.
3. **Phase 3 (Training)**: Identifies consecutive pairs with maximum corpus frequency, resolves ties lexicographically, and rewrites chunk tables.
4. **Phase 4 (Inference)**: Uses merge rank lookup to perform greedy left-to-right subword replacement.

---

## 📜 License

MIT License

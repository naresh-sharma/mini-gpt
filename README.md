# 🧠 MiniGPT: Learn by Building

**Build a GPT from scratch to understand how LLMs actually work.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/naresh-sharma/mini-gpt?style=social)](https://github.com/naresh-sharma/mini-gpt)

A hands-on series for software engineers who want to understand LLMs by building one. Each part ships as a [blog post on Async Thinking](https://asyncthinking.com/p/minigpt-learn-by-building), an interactive Colab notebook, and pure-Python source code you can read, run, and modify. Budget ~30-45 minutes per part.

**Prerequisites:** Python 3.10+ and basic programming. No machine learning background required.

## 📚 Series

| Part | Blog post | Notebook | Module |
|---|---|---|---|
| Intro: Why Build MiniGPT? | [post](https://asyncthinking.com/p/minigpt-learn-by-building) | — | — |
| **Part 1:** Tokenization | [post](https://asyncthinking.com/p/how-gpt-reads-your-words-and-why) | [open](notebooks/part1_tokenization.ipynb) | `mini_gpt.tokenizer` |
| **Part 2:** Embeddings | *coming soon* | [open](notebooks/part2_embeddings.ipynb) | `mini_gpt.embeddings` |
| Part 3: Attention | — | — | — |
| Part 4: Training | — | — | — |
| Part 5: Generation | — | — | — |

⭐ **Star the repo** to get notified when new parts drop.

## 💡 Quick example

```python
>>> from mini_gpt import SimpleTokenizer, TokenEmbedding
>>> from mini_gpt.utils import load_sample_vocab
>>>
>>> tokenizer = SimpleTokenizer(load_sample_vocab("simple"))
>>> ids = tokenizer.encode("Hello world!")
>>> ids
[6, 7, 8]
>>>
>>> emb = TokenEmbedding(vocab_size=1000, d_model=64, seed=42)
>>> emb.lookup(ids).shape
(3, 64)
```

Text → token IDs (Part 1) → dense vectors (Part 2). Attention, training, and generation follow.

## 🚀 Quick start

### Run in Colab (no setup)

- [Part 1: Tokenization](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb)
- [Part 2: Embeddings](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part2_embeddings.ipynb)

### Run locally

Requires Python 3.10+. On macOS the system `python3` is 3.9; use `brew install python@3.12` (or see [docs/INSTALL.md](docs/INSTALL.md)) and substitute `python3.12` in the venv step below.

```bash
git clone https://github.com/naresh-sharma/mini-gpt.git
cd mini-gpt

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

# Run a demo
python examples/embedding_demo.py   # Part 2: king - man + woman = queen

# Or open a notebook
jupyter notebook notebooks/
```

For more, see [docs/INSTALL.md](docs/INSTALL.md) (detailed setup), [docs/QUICKSTART.md](docs/QUICKSTART.md) (hands-on tour), [docs/FAQ.md](docs/FAQ.md) (troubleshooting), and [docs/ROADMAP.md](docs/ROADMAP.md) (what's planned).

## 📁 Project structure

```
mini-gpt/
├── notebooks/                    # One per part
│   ├── part1_tokenization.ipynb
│   └── part2_embeddings.ipynb
├── src/mini_gpt/                 # Reusable modules, one per part
│   ├── tokenizer.py                # Part 1
│   ├── embeddings.py               # Part 2
│   └── utils.py
├── examples/                     # Runnable demo scripts
│   ├── tokenizer_demo.py
│   ├── compare_with_tiktoken.py
│   ├── embedding_demo.py
│   └── visualize_embeddings.py
├── tests/                        # pytest unit tests
├── docs/                         # Install, FAQ, quickstart, roadmap
└── data/                         # Training data (Part 4+)
```

## 🤝 Contributing

Bug reports and pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). For questions or feedback, [open a discussion](https://github.com/naresh-sharma/mini-gpt/discussions) or [file an issue](https://github.com/naresh-sharma/mini-gpt/issues).

## 📞 Connect

[Async Thinking](https://asyncthinking.com) · [Twitter](https://x.com/Naresh_Sharma_) · [LinkedIn](https://linkedin.com/in/naresh-sharma-865b3b24/) · asyncthinking@gmail.com

## 📄 License

MIT — see [LICENSE](LICENSE).

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-04-15

Part 2 of the MiniGPT learning series: Embeddings.

### Added
- `TokenEmbedding`: (vocab_size, d_model) lookup with GPT-2 style random init (std=0.02). Supports 1D and batched (B, L) inputs — output shape is always `token_ids.shape + (d_model,)`.
- `SinusoidalPositionalEncoding`: deterministic sin/cos position signals from "Attention Is All You Need". No learned parameters; extrapolates past training lengths.
- `LearnedPositionalEmbedding`: GPT-2 style alternative, same mechanics as `TokenEmbedding` but indexed by position. Shares the `.encode(seq_len)` API with the sinusoidal variant so callers can swap them in one line.
- `build_input_embedding(token_ids, token_embedding, positional_encoder)`: helper that produces the `token_emb + pos_emb` sum attention will consume. Works on single sequences and batches.
- `notebooks/part2_embeddings.ipynb`: 38-cell interactive walkthrough — one-hot problem → NumPy lookup → torch.nn.Embedding equivalence → pretrained GloVe → `king - man + woman = queen` → PCA scatter → position demo → sinusoidal → learned → bridge to Part 3.
- `examples/embedding_demo.py`: runnable GloVe analogy demo.
- `examples/visualize_embeddings.py`: pure-NumPy PCA visualization of word clusters.
- 43 new tests in `tests/test_embeddings.py` covering shapes (1D, 2D, arbitrary rank), seed reproducibility, symmetry-breaking init, sinusoidal dot-product decay, positional broadcast across batch, and `d_model` mismatch guards.

### Changed
- `requirements.txt` organized with per-part section headers. Added `torch>=2.0.0`, `matplotlib>=3.7.0`, `gensim>=4.3.0` for Part 2's notebook and examples (the core module remains pure NumPy).
- README examples section grouped by part.

### Related
- Part 2 blog post on [Async Thinking](https://asyncthinking.com) — pending.

## [0.1.0] - 2025-10-27

Initial public release. Part 1 of the MiniGPT learning series.

### Added
- `SimpleTokenizer`: dictionary-based tokenizer with greedy longest-match encoding.
- `BPETokenizer`: educational Byte-Pair Encoding implementation (train, encode, decode, save/load).
- `CharacterTokenizer`: character-level tokenizer for baseline comparison.
- Tokenization utilities: visualization, tokenizer comparison, efficiency analysis, the "strawberry problem" demo.
- Sample vocabularies (`simple`, `english`, `code`) via `load_sample_vocab()`.
- `notebooks/part1_tokenization.ipynb`: interactive walkthrough.
- Example scripts: `tokenizer_demo.py`, `compare_with_tiktoken.py`.
- Test suite covering all tokenizers and utilities.
- GitHub Actions CI (tests + ruff).
- Contributing guide, code of conduct, issue and PR templates.

### Related
- Introduction post: [Introducing MiniGPT: Learn How LLMs Work by Building One](https://asyncthinking.com/p/minigpt-learn-by-building) (2025-10-28)
- Part 1 blog post: [How GPT Reads Your Words (And Why It Can't Count Letters)](https://asyncthinking.com/p/how-gpt-reads-your-words-and-why) (2025-11-24)

---

## Upcoming

- **v0.3.0** — Part 3: Attention (scaled dot-product, multi-head, causal self-attention).
- **v0.4.0** — Part 4: Training (loss, optimization, training loop on tiny Shakespeare).
- **v0.5.0** — Part 5: Generation (sampling strategies, temperature, top-k, top-p).

See [docs/ROADMAP.md](docs/ROADMAP.md) for details.

---

## Contributing to the Changelog

When making changes, add them to the `[Unreleased]` section under the appropriate category:
`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, or `Security`. When a release
is cut, move those entries under a new version heading with the release date.

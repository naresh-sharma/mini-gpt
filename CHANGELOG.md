# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- README, CHANGELOG, and ROADMAP updated to reflect Part 1 shipped status.

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

- **v0.2.0** — Part 2: Embeddings (token embeddings, positional encoding, visualization).
- **v0.3.0** — Part 3: Attention (scaled dot-product, multi-head, causal self-attention).
- **v0.4.0** — Part 4: Training (loss, optimization, training loop on tiny Shakespeare).
- **v0.5.0** — Part 5: Generation (sampling strategies, temperature, top-k, top-p).

See [docs/ROADMAP.md](docs/ROADMAP.md) for details.

---

## Contributing to the Changelog

When making changes, add them to the `[Unreleased]` section under the appropriate category:
`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, or `Security`. When a release
is cut, move those entries under a new version heading with the release date.

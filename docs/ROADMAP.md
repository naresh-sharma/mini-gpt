# Roadmap

Current plan for the MiniGPT learning series. Each part is a blog post on
[Async Thinking](https://asyncthinking.com) plus an interactive notebook and
working Python module in this repo.

## Current Status

- **Version**: 0.2.0
- **Last Updated**: 2026-04-15
- **Code shipped**: Parts 1–2
- **Blog posts published**: Introduction + Part 1 (Part 2 pending)

## Series Timeline

### Part 1: Tokenization ✅ Shipped
- [x] `SimpleTokenizer` (dictionary-based, greedy longest match)
- [x] `BPETokenizer` (Byte-Pair Encoding)
- [x] `CharacterTokenizer` (baseline comparison)
- [x] Tokenization utilities and visualizations
- [x] Interactive notebook
- [x] Blog post: [How GPT Reads Your Words](https://asyncthinking.com/p/how-gpt-reads-your-words-and-why)

### Part 2: Embeddings ✅ Code shipped
- [x] `TokenEmbedding` (pure-NumPy lookup with GPT-2 init, batched input support)
- [x] `SinusoidalPositionalEncoding` (deterministic sin/cos, dot-product decay)
- [x] `LearnedPositionalEmbedding` (GPT-2 style alternative, same `.encode(seq_len)` API)
- [x] `build_input_embedding` helper (token + positional, single & batched)
- [x] Visualizations (PCA scatter, sinusoidal heatmap, `king - man + woman ≈ queen`)
- [x] Interactive notebook
- [ ] Blog post

### Part 3: Attention
- [ ] Scaled dot-product attention
- [ ] Multi-head attention
- [ ] Causal self-attention (with look-ahead mask)
- [ ] Attention heatmap visualization
- [ ] Interactive notebook
- [ ] Blog post

### Part 4: Training
- [ ] Transformer block + full MiniGPT assembly
- [ ] Training loop on tiny Shakespeare
- [ ] Loss curves and training-progress visualization
- [ ] Interactive notebook
- [ ] Blog post

### Part 5: Generation
- [ ] Autoregressive generation loop
- [ ] Sampling strategies: greedy, temperature, top-k, top-p
- [ ] Interactive notebook
- [ ] Blog post
- [ ] Series wrap-up and PDF bundle

## Possible Follow-ups (Post-Series)

These are under consideration, not commitments:

- Part 6: Fine-tuning on a custom dataset
- Attention variants (Flash Attention, sliding window)
- Scaling laws, parameter efficiency
- RLHF basics
- Workshop / deep-dive course

## Feedback

- [GitHub Issues](https://github.com/naresh-sharma/mini-gpt/issues) for bugs and suggestions
- [GitHub Discussions](https://github.com/naresh-sharma/mini-gpt/discussions) for general feedback
- Email: asyncthinking@gmail.com

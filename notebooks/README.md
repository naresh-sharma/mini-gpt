# 📓 Notebooks

Interactive Jupyter notebooks for the MiniGPT series.

## 🚀 Quick Start

**Run in Google Colab (Recommended)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb)

**Or run locally:**
```bash
# Install MiniGPT
pip install -e .

# Start Jupyter
jupyter notebook
```

## 📚 Available Notebooks

### Part 1: Tokenization ✅
**How GPT Reads Your Words**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb)

**What you'll learn:**
- What tokenization is and why it matters
- Character-level vs word-level vs subword tokenization
- The famous "strawberry problem"
- Byte-Pair Encoding (BPE) - the algorithm used by GPT
- Hands-on implementation of tokenizers

**Prerequisites:** Basic Python knowledge

---

### Part 2: Embeddings 🚧
**Turning Words Into Meaning**

*Coming Soon*

**What you'll learn:**
- How tokens become vectors
- What embeddings are and why they matter
- How GPT understands word relationships
- Building your own embedding layer

---

### Part 3: Attention 🚧
**The Secret Sauce**

*Coming Soon*

**What you'll learn:**
- How attention mechanisms work
- Self-attention and multi-head attention
- Why attention is so powerful
- Building attention from scratch

---

### Part 4: Model Architecture 🚧
**From Random to Shakespeare**

*Coming Soon*

**What you'll learn:**
- How all the pieces fit together
- Transformer architecture
- Training a mini-GPT model
- Understanding the full pipeline

---

### Part 5: Training & Generation 🚧
**Making Your GPT Talk**

*Coming Soon*

**What you'll learn:**
- How to train your model
- Text generation strategies
- Fine-tuning techniques
- Deploying your model

---

## 🛠 Alternative: Explore the Code

If you prefer to explore the code directly:

### 🚀 Option 1: Python Demo Scripts
```bash
# Run the interactive demo
python examples/tokenizer_demo.py

# See all examples
ls examples/
```

### 💻 Option 2: Use the Package Directly
```bash
# Install the package
pip install -e .

# Use in Python or IPython
python
```
```python
from mini_gpt import SimpleTokenizer, BPETokenizer
from mini_gpt.utils import load_sample_vocab, visualize_tokens

# Create a tokenizer
vocab = load_sample_vocab("simple")
tokenizer = SimpleTokenizer(vocab)

# Tokenize text
text = "Hello world!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Visualize
visualize_tokens(text, tokenizer)
```

### 📖 Option 3: Read the Blog Post

For detailed explanations with visuals:
- [Part 1: How GPT Reads Your Words](https://asyncthinking.com) (Coming Soon)

## 📅 Notebook Roadmap

| Part | Topic | Status | Expected |
|------|-------|--------|----------|
| 1 | **Tokenization** | 🚧 In Progress | Week of [Date] |
| 2 | **Embeddings** | 📅 Planned | TBD |
| 3 | **Attention** | 📅 Planned | TBD |
| 4 | **Training** | 📅 Planned | TBD |
| 5 | **Generation** | 📅 Planned | TBD |

**⭐ Star the repo to get notified when notebooks drop!**

## 🤝 Want to Help?

Interested in contributing to the notebooks? 

- Check out our [Contributing Guide](../CONTRIBUTING.md)
- Share what you'd like to see in the notebooks via [Discussions](https://github.com/naresh-sharma/mini-gpt/discussions)

## 💡 Why Python Examples First?

We're prioritizing **working code** over notebooks because:
1. Python scripts work everywhere (no Colab needed)
2. Easy to run and modify locally
3. Better for understanding the actual implementation
4. Notebooks will complement the code, not replace it

Think of it as: **Code first, notebooks second** 🚀

## 📞 Questions?

- 💬 [Start a Discussion](https://github.com/naresh-sharma/mini-gpt/discussions)
- 🐛 [Report an Issue](https://github.com/naresh-sharma/mini-gpt/issues)
- 📧 Email: asyncthinking@gmail.com

---

*Notebooks are coming soon! In the meantime, the Python examples provide a fully interactive learning experience.*
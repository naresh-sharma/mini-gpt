# 🧠 MiniGPT: Learn by Building

[![Read the Introduction](https://img.shields.io/badge/Read-Introduction-blue)](https://asyncthinking.com/p/minigpt-learn-by-building)
[![GitHub stars](https://img.shields.io/github/stars/naresh-sharma/mini-gpt?style=social)](https://github.com/naresh-sharma/mini-gpt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb)

**A hands-on series: Build a GPT from scratch and finally understand how LLMs actually work**

🚀 **[Read the full introduction →](https://asyncthinking.com/p/minigpt-learn-by-building)**

## 🎯 What You'll Learn

- **Tokenization**: How text becomes numbers
- **Embeddings**: Converting tokens to meaningful vectors  
- **Attention**: The mechanism that makes transformers powerful
- **Model Architecture**: Building the complete GPT structure
- **Training**: Teaching your model to predict the next token

## 💡 Quick Example

```python
>>> from mini_gpt import SimpleTokenizer
>>> from mini_gpt.utils import load_sample_vocab
>>> vocab = load_sample_vocab("simple")
>>> tokenizer = SimpleTokenizer(vocab)
>>> tokenizer.encode("Hello world!")
[6, 7, 8]
```

*Learn why it works this way in Part 1!*

## 🤔 Why Build From Scratch?

You've used ChatGPT. Maybe you've even fine-tuned a model. But do you really understand:
- Why GPT can't count letters reliably?
- Why context windows have token limits?
- What "attention" actually computes?
- Why prompt engineering even works?

**Building from scratch is the only way to truly understand.**

This series takes you from "I can use GPT" to "I understand how GPT works."

## 📚 Series

- **[Introduction: Why Build MiniGPT?](https://asyncthinking.com/p/minigpt-learn-by-building)** ✅ Published
- **[Part 1: Tokenization](https://asyncthinking.com/p/how-gpt-reads-your-words-and-why)** ✅ Published
- Part 2: Embeddings
- Part 3: Attention Mechanism
- Part 4: Transformer Architecture
- Part 5: Training and Generation
- Part 6: Fine-tuning and Prompt Engineering

**🔔 Star the repo to get notified when new parts drop!**

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
**Run Part 1: Tokenization in Colab**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb)

**Why Colab?**
- ✅ **No setup required** - runs in your browser
- ✅ **Free GPU access** - for future parts
- ✅ **Pre-installed libraries** - everything works out of the box
- ✅ **Shareable notebooks** - easy to collaborate

> 💡 **Need help setting up?**  
> See [📖 Installation Guide](docs/INSTALL.md) or [⚡ Quick Start Guide](docs/QUICKSTART.md).

#### 🆘 Colab Troubleshooting

**If the notebook doesn't load:**
1. **Refresh the page** - Colab sometimes has loading issues
2. **Try the direct link** - [part1_tokenization.ipynb](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb)
3. **Check your internet** - Colab needs a stable connection

**If you get import errors:**
1. **Run the first cell** - It installs MiniGPT automatically
2. **Restart runtime** - Go to Runtime → Restart runtime
3. **Run all cells** - Click Runtime → Run all

**If the notebook is slow:**
1. **Enable GPU** - Runtime → Change runtime type → GPU (for future parts)
2. **Clear outputs** - Edit → Clear all outputs
3. **Restart runtime** - Runtime → Restart runtime

### Option 2: Local Installation

#### Step 1: Set Up Virtual Environment (Recommended)

**Why use a virtual environment?** It keeps your project dependencies isolated and prevents conflicts with other Python projects.

<details>
<summary><b>🐍 Create Virtual Environment (Click to expand)</b></summary>

**On macOS/Linux:**
```bash
# Navigate to project directory
cd mini-gpt

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

**On Windows:**
```bash
# Navigate to project directory
cd mini-gpt

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

**To deactivate later:**
```bash
deactivate
```

</details>

#### Step 2: Install MiniGPT
```bash
# Make sure your virtual environment is activated!
# You should see (venv) in your prompt

# Install dependencies
pip install -r requirements.txt

# Install MiniGPT in development mode
pip install -e .

# Verify installation
python -c "from mini_gpt import SimpleTokenizer; print('✅ Installation successful!')"
```

#### Step 3: Try It Out
```bash
# Run the demo
python examples/tokenizer_demo.py

# Or start Python and experiment
python
>>> from mini_gpt import SimpleTokenizer
>>> from mini_gpt.utils import load_sample_vocab
>>> vocab = load_sample_vocab("simple")
>>> tokenizer = SimpleTokenizer(vocab)
>>> print(tokenizer.encode("Hello world!"))
[6, 7, 8]
```

### Option 3: Pip Install (Coming Soon)
```bash
pip install mini-gpt
```
*PyPI package coming soon! For now, clone and install locally using the steps above.*

## 💡 Examples

Try these example scripts to see MiniGPT in action:

```bash
# Interactive tokenizer demo
python examples/tokenizer_demo.py

# Basic usage examples  
python examples/basic_usage.py

# Compare with OpenAI's tokenizer
python examples/compare_with_tiktoken.py
```

## 👥 Who Is This For?

- **Software engineers** curious about AI
- **Developers** who learn by building  
- **Students** seeking hands-on ML experience
- **Anyone** who wants to understand GPT, not just use it

**Prerequisites**: Basic Python knowledge. No machine learning background required!

## ✨ What Makes This Different?

| Traditional ML Courses | MiniGPT Series |
|----------------------|----------------|
| Heavy on theory, light on code | **Code-first approach** |
| Assumes ML background | **No ML experience needed** |
| Abstract mathematical concepts | **Concrete Python implementations** |
| Large, complex datasets | **Simple, understandable examples** |
| Black box explanations | **Transparent, step-by-step building** |

## 📁 Project Structure

```
mini-gpt/
├── 📓 notebooks/           # Interactive Jupyter notebooks (one per part as it ships)
│   └── part1_tokenization.ipynb
├── 🐍 src/mini_gpt/        # Reusable Python modules (added per part as it ships)
│   ├── tokenizer.py
│   └── utils.py
├── 📊 data/                # Training datasets (populated when Part 4 ships)
├── 🧪 tests/               # Unit tests
├── 📚 docs/                # Additional documentation
└── 💡 examples/            # Example scripts
    ├── tokenizer_demo.py          # Interactive tokenizer demo
    └── compare_with_tiktoken.py   # Compare with OpenAI's tokenizer
```

## 🎯 Code Philosophy

- **Readable over clever**: Code should teach, not impress
- **Simple over optimized**: Understanding first, performance second
- **Commented over concise**: Every important line explained
- **Modular over monolithic**: Each concept in its own file
- **Educational over production**: Built for learning, not scaling

## 📋 Prerequisites

**Required:**
- Python 3.8+ installed
- Basic Python knowledge (functions, classes, loops)
- A curious mind! 🤔

**Helpful but not required:**
- Jupyter/Colab experience
- Basic linear algebra (vectors, matrices)
- NumPy familiarity

**NOT required:**
- Machine learning background
- Advanced mathematics
- GPU or powerful hardware

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

- 🐛 Found a bug? [Report it](https://github.com/naresh-sharma/mini-gpt/issues)
- 💡 Have an idea? [Suggest a feature](https://github.com/naresh-sharma/mini-gpt/issues)
- 📝 Want to improve docs? [Submit a PR](https://github.com/naresh-sharma/mini-gpt/pulls)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Connect

- 📧 Email: asyncthinking@gmail.com
- 🐦 Twitter: [@Naresh_Sharma_](https://x.com/Naresh_Sharma_)
- 💼 LinkedIn: [Naresh Sharma](https://linkedin.com/in/naresh-sharma-865b3b24/)
- 🌐 Blog: [Async Thinking](https://asyncthinking.com)

## 🚀 Ready to Start?

**Three ways to begin:**

1. **🔥 Jump right in** → [Open Part 1 in Colab](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb) (no setup required)
2. **📖 Read first** → [Async Thinking Blog](https://asyncthinking.com)
3. **⭐ Star the repo** → Get notified when new parts drop

**Time investment:** ~30-45 minutes per part. By the end, you'll have built a working GPT!

---

### 💬 Questions?

- 🐛 [Report issues](https://github.com/naresh-sharma/mini-gpt/issues)
- 💡 [Request features](https://github.com/naresh-sharma/mini-gpt/issues)
- 💬 [Join discussions](https://github.com/naresh-sharma/mini-gpt/discussions)
- 📧 Email: asyncthinking@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/naresh-sharma-865b3b24/)

**Let's demystify AI together!** 🧠✨

---

**MiniGPT © 2025 Naresh Sharma**  
Licensed under [MIT](LICENSE) • [GitHub Repo](https://github.com/naresh-sharma/mini-gpt)

*Made with ❤️ for the curious developer*

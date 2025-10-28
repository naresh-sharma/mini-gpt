# 🧠 MiniGPT: Learn by Building

[![GitHub stars](https://img.shields.io/github/stars/naresh-sharma/mini-gpt?style=social)](https://github.com/naresh-sharma/mini-gpt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb)

> **If you can read Python, you can understand GPT** 🧠

**MiniGPT** — a hands-on project to build and understand GPT models step by step. Build a miniature GPT from scratch to understand how Large Language Models work. This educational series is designed for software engineers who want to demystify the magic behind modern AI.

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

## 📚 The Series

Build GPT step-by-step across 5 hands-on parts:

| Part | What You'll Build | Read | Code | Status |
|------|------------------|------|------|--------|
| **1** | **How GPT Reads Your Words**<br>Build a tokenizer and understand why GPT can't spell | [📖 Blog →](https://asyncthinking.com) | [💻 Colab →](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb) | ✅ **Live** |
| **2** | **Turning Words into Meaning**<br>Create embeddings and visualize semantic space | [📖 Blog →](https://asyncthinking.com) | [💻 Colab →](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part2_embeddings.ipynb) | 🧠 **In Progress** |
| **3** | **The Secret Sauce**<br>Implement attention from scratch | [📖 Blog →](https://asyncthinking.com) | [💻 Colab →](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part3_attention.ipynb) | ⏳ **Coming Soon** |
| **4** | **From Random to Shakespeare**<br>Train your GPT on real text | [📖 Blog →](https://asyncthinking.com) | [💻 Colab →](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part4_model.ipynb) | ⏳ **Coming Soon** |
| **5** | **Making Your GPT Talk**<br>Generate text and understand its quirks | [📖 Blog →](https://asyncthinking.com) | [💻 Colab →](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part5_training.ipynb) | ⏳ **Coming Soon** |

**⏱️ Time commitment:** ~30-45 min per part • **Total:** 3-4 hours to complete

**📝 Note:** Blog posts are being written on [Async Thinking](https://asyncthinking.com) and will be linked here when ready.

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
├── 📓 notebooks/           # Interactive Jupyter notebooks
│   ├── part1_tokenization.ipynb
│   ├── part2_embeddings.ipynb
│   ├── part3_attention.ipynb
│   ├── part4_model.ipynb
│   └── part5_training.ipynb
├── 🐍 src/mini_gpt/        # Reusable Python modules
│   ├── tokenizer.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── model.py
│   ├── trainer.py
│   └── generator.py
├── 📊 data/                # Training datasets
│   └── tiny_shakespeare.txt
├── 🎨 visuals/             # Diagrams and visualizations
├── 🧪 tests/               # Unit tests
├── 📚 docs/                # Additional documentation
└── 💡 examples/            # Example scripts
    ├── tokenizer_demo.py   # Interactive tokenizer demo
    ├── basic_usage.py      # Simple usage examples
    └── compare_with_tiktoken.py  # Compare with OpenAI's tokenizer
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

- 📧 Email: nareshsharman@gmail.com
- 🐦 Twitter: [@Naresh_Sharma_](https://x.com/Naresh_Sharma_)
- 💼 LinkedIn: [Naresh Sharma](https://linkedin.com/in/naresh-sharma-865b3b24/)
- 🌐 Blog: [Async Thinking](https://asyncthinking.com)

## 🚀 Ready to Start?

**Three ways to begin:**

1. **🔥 Jump right in** → [Open Part 1 in Colab](https://colab.research.google.com/github/naresh-sharma/mini-gpt/blob/main/notebooks/part1_tokenization.ipynb) (no setup required)
2. **📖 Read first** → [Async Thinking Blog](https://asyncthinking.com) (blog posts coming soon)
3. **⭐ Star the repo** → Get notified when new parts drop

**Time investment:** ~30-45 minutes per part. By the end, you'll have built a working GPT!

---

### 💬 Questions?

- 🐛 [Report issues](https://github.com/naresh-sharma/mini-gpt/issues)
- 💡 [Request features](https://github.com/naresh-sharma/mini-gpt/issues)
- 💬 [Join discussions](https://github.com/naresh-sharma/mini-gpt/discussions)
- 📧 Email: nareshsharman@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/naresh-sharma-865b3b24/)

**Let's demystify AI together!** 🧠✨

---

**MiniGPT © 2025 Naresh Sharma**  
Licensed under [MIT](LICENSE) • [GitHub Repo](https://github.com/naresh-sharma/mini-gpt)

*Made with ❤️ for the curious developer*

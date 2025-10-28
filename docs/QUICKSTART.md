# ⚡ Quick Start Guide

Get MiniGPT running in 5 minutes!

## 📋 Checklist

Before you start, make sure you have:
- [ ] Python 3.8 or higher installed
- [ ] Terminal/Command Prompt access
- [ ] 10 minutes of time

Not sure? Check Python version: `python --version` or `python3 --version`

---

## 🚀 5-Minute Setup

### 1️⃣ Get the Code (1 min)
```bash
# Clone the repository
git clone https://github.com/naresh-sharma/mini-gpt.git
cd mini-gpt
```

### 2️⃣ Create Virtual Environment (1 min)

**Copy and paste this:**
```bash
# macOS/Linux:
python3 -m venv venv && source venv/bin/activate

# Windows:
python -m venv venv && venv\Scripts\activate
```

**You should see `(venv)` appear in your prompt!**

### 3️⃣ Install MiniGPT (2 min)
```bash
pip install -r requirements.txt
pip install -e .
```

### 4️⃣ Try It! (1 min)
```bash
python examples/tokenizer_demo.py
```

**You should see tokenization in action! 🎉**

---

## 🎓 Next Steps

Now that it's working:

1. **Read Part 1**: [How GPT Reads Your Words](https://github.com/naresh-sharma/mini-gpt#-the-series)
2. **Explore the code**: Open `src/mini_gpt/tokenizer.py`
3. **Experiment**: Try tokenizing your own text!
```python
python
>>> from mini_gpt import SimpleTokenizer
>>> from mini_gpt.utils import load_sample_vocab
>>> vocab = load_sample_vocab("simple")
>>> tokenizer = SimpleTokenizer(vocab)
>>> tokenizer.encode("Your text here!")
```

---

## 🆘 Something Broke?

**Common fixes:**

| Problem | Solution |
|---------|----------|
| `python: command not found` | Try `python3` instead |
| `pip: command not found` | Try `pip3` or `python -m pip` |
| Import errors | Make sure `(venv)` is in your prompt |
| Permission errors | Never use `sudo` with pip in venv |

**Still stuck?** 
- [Check the full installation guide](INSTALL.md)
- [Ask for help](https://github.com/naresh-sharma/mini-gpt/discussions)

---

## 💡 Remember

**Every time you open a new terminal:**
```bash
cd mini-gpt
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

Then you're ready to work! 🚀

# 📦 Installation Guide

Complete installation instructions for MiniGPT.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning)

**Check your Python version:**
```bash
python --version
# or
python3 --version
```

---

## 🎯 Installation Methods

### Method 1: Local Development Setup (Recommended)

This method is best if you want to:
- Explore and modify the code
- Contribute to the project
- Learn by experimenting

#### Step 1: Get the Code

**Option A: Clone with Git**
```bash
git clone https://github.com/naresh-sharma/mini-gpt.git
cd mini-gpt
```

**Option B: Download ZIP**
1. Go to https://github.com/naresh-sharma/mini-gpt
2. Click "Code" → "Download ZIP"
3. Extract and navigate to the folder
```bash
cd mini-gpt
```

#### Step 2: Create Virtual Environment

**What is a virtual environment?**
A virtual environment is an isolated Python environment that keeps your project dependencies separate from your system Python. This prevents version conflicts and keeps things clean.

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your prompt should now show (venv)
```

**On Windows (Command Prompt):**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate.bat

# Your prompt should now show (venv)
```

**On Windows (PowerShell):**
```bash
# Create virtual environment
python -m venv venv

# Activate it (you may need to enable scripts first)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1

# Your prompt should now show (venv)
```

**Troubleshooting Activation:**
- If you get "command not found" on macOS/Linux, try `python3 -m venv venv`
- If you get permission errors on Windows PowerShell, run as Administrator
- Make sure you're in the `mini-gpt` directory

#### Step 3: Install Dependencies
```bash
# Make sure virtual environment is activated!
# You should see (venv) in your prompt

# Upgrade pip (recommended)
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install MiniGPT in editable/development mode
pip install -e .
```

**What does `pip install -e .` do?**
The `-e` flag installs the package in "editable" mode, meaning changes you make to the code are immediately reflected without reinstalling.

#### Step 4: Verify Installation
```bash
# Test import
python -c "from mini_gpt import SimpleTokenizer; print('✅ Success!')"

# Run the demo
python examples/tokenizer_demo.py
```

If everything works, you're ready to go! 🎉

---

### Method 2: Quick Install (For Using the Package)

If you just want to use MiniGPT without modifying the code:
```bash
# Create and activate virtual environment (see Step 2 above)
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install from GitHub
pip install git+https://github.com/naresh-sharma/mini-gpt.git

# Verify
python -c "from mini_gpt import SimpleTokenizer; print('✅ Installed!')"
```

---

### Method 3: PyPI Install (Coming Soon)

Once published to PyPI:
```bash
pip install mini-gpt
```

---

## 🧹 Managing Your Virtual Environment

### Activating

You need to activate the virtual environment every time you open a new terminal:
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Tip:** Add an alias to your shell config:
```bash
# In ~/.bashrc or ~/.zshrc
alias venv='source venv/bin/activate'

# Then just run:
venv
```

### Deactivating

When you're done working:
```bash
deactivate
```

### Deleting

To remove the virtual environment:
```bash
# Make sure it's deactivated first
deactivate

# Delete the folder
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows
```

---

## 🔧 Troubleshooting

### "pip: command not found"

**Solution:**
```bash
# Try pip3 instead
pip3 install -r requirements.txt

# Or use Python's module syntax
python -m pip install -r requirements.txt
python3 -m pip install -r requirements.txt
```

### "python: command not found"

**Solution:**
```bash
# Try python3
python3 --version
python3 -m venv venv
```

### Virtual Environment Won't Activate on Windows

**Solution for PowerShell:**
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
venv\Scripts\Activate.ps1
```

### Import Error: "No module named 'mini_gpt'"

**Solutions:**
1. Make sure virtual environment is activated (you should see `(venv)` in prompt)
2. Run `pip install -e .` in the project root
3. Check you're in the correct directory: `pwd` (should show `.../mini-gpt`)

### "Permission denied" errors

**Solution:**
```bash
# Don't use sudo with pip in virtual environments!
# Instead, make sure venv is activated

# If you used sudo before, remove and reinstall:
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 🐳 Alternative: Docker (Advanced)

For a completely isolated environment:
```bash
# Coming soon!
# docker run -it mini-gpt:latest
```

---

## 💡 Best Practices

### DO:
✅ Always use a virtual environment for Python projects
✅ Activate the venv before installing packages
✅ Keep requirements.txt updated
✅ Use `pip list` to see installed packages

### DON'T:
❌ Install packages globally with sudo
❌ Mix system Python with virtual environment
❌ Commit the `venv/` folder to Git (it's in .gitignore)
❌ Share your venv folder between projects

---

## 📚 Learn More

- [Official Python venv Documentation](https://docs.python.org/3/library/venv.html)
- [Real Python: Virtual Environments Primer](https://realpython.com/python-virtual-environments-a-primer/)
- [Why Use Virtual Environments?](https://realpython.com/python-virtual-environments-a-primer/#why-do-you-need-virtual-environments)

---

## ❓ Still Having Issues?

- 💬 [Ask in Discussions](https://github.com/naresh-sharma/mini-gpt/discussions)
- 🐛 [Report a Bug](https://github.com/naresh-sharma/mini-gpt/issues)
- 📧 Email: asyncthinking@gmail.com

Include:
- Your OS (Windows/Mac/Linux)
- Python version (`python --version`)
- Error message (full output)
- Steps you've tried

#!/bin/bash
# MiniGPT Part 1 Readiness Check Script

echo "🔍 Checking if MiniGPT Part 1 is ready for first commit..."
echo ""

# Check critical files exist
files=(
    "README.md"
    "LICENSE"
    "setup.py"
    "requirements.txt"
    "src/mini_gpt/__init__.py"
    "src/mini_gpt/tokenizer.py"
    "src/mini_gpt/utils.py"
    "examples/tokenizer_demo.py"
    ".gitignore"
    "MANIFEST.in"
)

missing=0
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - MISSING"
        missing=$((missing + 1))
    fi
done

echo ""

# Test import
echo "🧪 Testing package imports..."
python -c "from mini_gpt import SimpleTokenizer, BPETokenizer; print('✅ Package imports work')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Package imports work"
else
    echo "❌ Package imports FAIL"
    missing=$((missing + 1))
fi

# Test demo
echo "🧪 Testing demo script..."
python examples/tokenizer_demo.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Demo script works"
else
    echo "⚠️  Demo script has issues"
fi

# Test core functionality
echo "🧪 Testing core functionality..."
python -c "
import sys; sys.path.insert(0, 'src')
from mini_gpt.tokenizer import SimpleTokenizer
from mini_gpt.utils import load_sample_vocab
vocab = load_sample_vocab('simple')
tokenizer = SimpleTokenizer(vocab)
tokens = tokenizer.encode('Hello world!')
decoded = tokenizer.decode(tokens)
assert decoded == 'Hello world!'
print('✅ Core tokenization works')
" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Core functionality works"
else
    echo "❌ Core functionality FAIL"
    missing=$((missing + 1))
fi

echo ""
echo "================================"
if [ $missing -eq 0 ]; then
    echo "🎉 READY TO COMMIT!"
    echo ""
    echo "Next steps:"
    echo "1. git add ."
    echo "2. git commit -m 'Initial commit: Part 1 - Tokenization'"
    echo "3. git push origin main"
    echo ""
    echo "Your MiniGPT Part 1 is ready! 🚀"
else
    echo "⚠️  NOT READY - Fix $missing critical issues first"
    echo ""
    echo "Missing files or functionality need to be addressed."
fi
echo "================================"

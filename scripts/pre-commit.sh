#!/bin/bash
# Pre-commit script for MiniGPT
# Run all quality checks before committing

set -e

echo "🧠 MiniGPT Pre-commit Checks"
echo "=============================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Run pre-commit hooks
echo "🔍 Running pre-commit hooks..."
pre-commit run --all-files

echo ""
echo "✅ All checks passed! Ready to commit."
echo ""
echo "💡 Tip: You can also run individual checks:"
echo "   ruff check .                    # Lint code"
echo "   ruff format .                   # Format code"
echo "   pytest tests/ -v               # Run tests"
echo ""
echo "🎯 This project uses Ruff for all linting and formatting!"

# Contributing to MiniGPT 🚀

Thank you for your interest in contributing to MiniGPT! This project is designed to be educational and accessible, so we welcome contributions from developers of all skill levels.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Guidelines](#documentation-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## 🤝 Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to nareshsharman@gmail.com.

## 🎯 How Can I Contribute?

### 🐛 Bug Reports
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Check existing issues first
- Provide clear reproduction steps
- Include environment details

### 💡 Feature Requests
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the problem you're trying to solve
- Describe your proposed solution
- Consider alternatives you've thought about

### 📝 Documentation Improvements
- Fix typos and grammar
- Improve clarity and examples
- Add missing explanations
- Update outdated information

### 🧪 Code Contributions
- Fix bugs
- Implement new features
- Improve performance
- Add tests
- Refactor for clarity

## 🎯 Your First Contribution

Never contributed to open source before? No problem! Here are some good first issues:

1. **Fix typos in documentation** (look for `good-first-issue` label)
2. **Add more examples to notebooks**
3. **Improve error messages**
4. **Write additional tests**

**Step-by-step first PR:**
1. Find an issue labeled `good-first-issue`
2. Comment "I'd like to work on this"
3. Fork the repo
4. Make your changes
5. Submit a PR
6. Respond to feedback
7. Celebrate! 🎉

**We're here to help!** Don't hesitate to ask questions in the issue.

## 🛠 Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)
- A code editor (VS Code recommended)

### Setup Steps

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/naresh-sharma/mini-gpt.git
   cd mini-gpt
   ```

2. **Create a virtual environment**
   ```bash
   # Create venv
   python3 -m venv venv
   
   # Activate it
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   # Upgrade pip
   pip install --upgrade pip
   
   # Install requirements
   pip install -r requirements.txt
   
   # Install in development mode
   pip install -e .
   ```

4. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

6. **Set up pre-commit hooks (recommended)**
   ```bash
   # Install pre-commit
   pip install pre-commit
   
   # Install hooks
   pre-commit install
   
   # Run on all files (optional)
   pre-commit run --all-files
   ```

**Pro tip:** Always activate your venv before working! You should see `(venv)` in your terminal prompt.

### 🔍 Pre-commit Checks

This project uses pre-commit hooks to ensure code quality. The hooks run automatically before each commit and include:

- **Ruff**: Linting and formatting
- **MyPy**: Type checking  
- **Pytest**: Running tests

**Manual checks:**
```bash
# Quick way - run all checks
./scripts/pre-commit.sh

# Or manually
pre-commit run --all-files

# Individual checks
ruff check .                    # Lint code
ruff format .                   # Format code  
mypy src/ --ignore-missing-imports  # Type check
pytest tests/ -v               # Run tests
```

## 🔄 Pull Request Process

### Before Submitting
- [ ] Fork the repository
- [ ] Create a feature branch (`git checkout -b feature/amazing-feature`)
- [ ] Make your changes
- [ ] Add tests for new functionality
- [ ] Ensure all tests pass (`pytest`)
- [ ] Run code formatting (`black .` and `isort .`)
- [ ] Update documentation if needed
- [ ] Commit your changes (`git commit -m 'Add amazing feature'`)
- [ ] Push to your branch (`git push origin feature/amazing-feature`)

### Pull Request Guidelines
- Use a clear, descriptive title
- Reference any related issues
- Provide a detailed description of changes
- Include screenshots for UI changes
- Ensure CI checks pass
- Request review from maintainers

### Review Process
- All PRs require review before merging
- Address feedback promptly
- Keep PRs focused and reasonably sized
- Update documentation as needed

## 🎨 Code Style Guidelines

### Python Code
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length: 100 characters
- Use type hints where appropriate

### Formatting Commands
```bash
# Format code
black .

# Sort imports
isort .

# Check code style
flake8 .

# Type checking
mypy src/
```

### Naming Conventions
- **Functions and variables**: `snake_case`
  ```python
  def calculate_token_count(text: str) -> int:
      token_count = len(text.split())
      return token_count
  ```
- **Classes**: `PascalCase`
  ```python
  class SimpleTokenizer:
      def __init__(self, vocab: Dict[str, int]):
          self.vocab = vocab
  ```
- **Constants**: `UPPER_SNAKE_CASE`
  ```python
  MAX_SEQUENCE_LENGTH = 1024
  DEFAULT_VOCAB_SIZE = 1000
  ```
- **Private methods**: `_leading_underscore`
  ```python
  def _create_positional_encoding(self) -> torch.Tensor:
      # Internal helper method
      pass
  ```

### Documentation
- Use Google-style docstrings
- Include type hints in function signatures
- Add comments for complex logic
- Keep docstrings up to date

## 🧪 Testing Requirements

### Test Coverage
- Aim for >80% test coverage
- Write tests for new features
- Include edge cases and error conditions
- Test both success and failure paths

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mini_gpt

# Run specific test file
pytest tests/test_tokenizer.py

# Run with verbose output
pytest -v
```

### Test Guidelines
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Use fixtures for common setup
- Mock external dependencies
- Test both positive and negative cases

## 📚 Documentation Guidelines

### README Updates
- Keep installation instructions current
- Update feature lists
- Maintain accurate examples
- Check all links work

### Code Documentation
- Document all public functions and classes
- Include parameter descriptions
- Provide usage examples
- Explain complex algorithms

### Notebook Documentation
- Add markdown cells explaining concepts
- Include visualizations where helpful
- Provide clear step-by-step instructions
- Test all code cells

## 🐛 Reporting Bugs

### Before Reporting
1. Check if the bug has already been reported
2. Try to reproduce the issue
3. Check the latest version
4. Gather relevant information

### Bug Report Template
Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) or include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed reproduction steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, package versions
- **Additional Context**: Screenshots, logs, etc.

## 💡 Suggesting Features

### Before Suggesting
1. Check existing feature requests
2. Consider if it fits the project's educational goals
3. Think about implementation complexity
4. Consider impact on learning experience

### Feature Request Template
Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) or include:

- **Problem**: What problem does this solve?
- **Solution**: Describe your proposed solution
- **Alternatives**: Other solutions you've considered
- **Additional Context**: Any other relevant information

## 🏷 Labels and Milestones

### Issue Labels
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information is requested

### Milestones
- `v0.1.0`: Initial release
- `v0.2.0`: Part 2 completion
- `v0.3.0`: Part 3 completion
- `v0.4.0`: Part 4 completion
- `v0.5.0`: Part 5 completion

## 🎓 Educational Focus

Remember: This project is educational! When contributing:

- **Prioritize clarity over cleverness**
- **Add comments explaining complex concepts**
- **Include examples and visualizations**
- **Consider the learning journey**
- **Make it accessible to beginners**

## 📞 Getting Help

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Use Issues for bugs and feature requests
- 📧 **Email**: nareshsharman@gmail.com for private matters
- 💼 **LinkedIn**: [Naresh Sharma](https://linkedin.com/in/naresh-sharma-865b3b24/)

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for helping make MiniGPT better! 🚀

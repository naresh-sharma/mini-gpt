"""
MiniGPT: Learn by Building

A hands-on series for understanding Large Language Models. This package
grows part-by-part — each shipped part adds its own module here.

Shipped:
- Part 1: Tokenization (SimpleTokenizer, BPETokenizer)

Coming:
- Part 2: Embeddings
- Part 3: Attention
- Part 4: Training
- Part 5: Generation
"""

from .tokenizer import BPETokenizer, SimpleTokenizer

__version__ = "0.1.0"
__author__ = "Naresh Sharma"
__email__ = "asyncthinking@gmail.com"
__description__ = "Build GPT from scratch to understand how LLMs work"
__license__ = "MIT"

__all__ = [
    "__author__",
    "__description__",
    "__email__",
    "__license__",
    "__version__",
    "BPETokenizer",
    "SimpleTokenizer",
]

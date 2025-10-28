"""
MiniGPT: Learn by Building
A hands-on series for understanding Large Language Models

This package provides educational implementations of GPT components
to help developers understand how Large Language Models work.

Part 1: Tokenization - How GPT reads your words
Part 2: Embeddings - Turning words into meaning
Part 3: Attention - The secret sauce
Part 4: Model Architecture - From random to Shakespeare
Part 5: Training & Generation - Making your GPT talk
"""

__version__ = "0.1.0"
__author__ = "Naresh Sharma"
__email__ = "[your-email@example.com]"
__description__ = "Build GPT from scratch to understand how LLMs work"
__license__ = "MIT"

# Part 1: Tokenization (implemented)
from .tokenizer import BPETokenizer, SimpleTokenizer

# TODO: Add imports as modules are implemented
# from .embeddings import EmbeddingLayer
# from .attention import MultiHeadAttention
# from .model import MiniGPT
# from .trainer import Trainer
# from .generator import Generator

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__license__",
    # Part 1: Tokenization
    "SimpleTokenizer",
    "BPETokenizer",
    # TODO: Add other components as implemented
    # "EmbeddingLayer",
    # "MultiHeadAttention",
    # "MiniGPT",
    # "Trainer",
    # "Generator",
]

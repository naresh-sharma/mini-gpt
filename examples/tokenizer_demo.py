#!/usr/bin/env python3
"""
MiniGPT Tokenizer Demo

Demonstrates how to use the SimpleTokenizer and BPETokenizer.
Run this script to see tokenization in action!

Usage:
    python examples/tokenizer_demo.py
"""

from mini_gpt.tokenizer import BPETokenizer, SimpleTokenizer
from mini_gpt.utils import (
    demonstrate_strawberry_problem,
    load_sample_vocab,
    visualize_tokens,
)


def demo_simple_tokenizer():
    """Demonstrate SimpleTokenizer usage"""
    print("=" * 60)
    print("SIMPLE TOKENIZER DEMO")
    print("=" * 60)
    print("The SimpleTokenizer uses a dictionary-based approach.")
    print("It demonstrates the core idea: text → tokens → IDs")
    print()

    # Create vocab with some interesting cases
    vocab = {
        "Hello": 1,
        " world": 2,  # Note: space at beginning
        "!": 3,
        "?": 4,
        "straw": 5,
        "berry": 6,
        "strawberry": 7,  # This will be preferred over "straw" + "berry"
        "ing": 8,
        "ed": 9,
        "er": 10,
    }

    tokenizer = SimpleTokenizer(vocab)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print()

    # Test cases
    test_texts = [
        "Hello world!",
        "strawberry",  # Should use "strawberry" not "straw" + "berry"
        "strawberries",  # Will need to use "straw" + "berry" + "s" (unknown)
        "Hello there!",  # "there" is unknown
    ]

    for text in test_texts:
        print(f"Text: '{text}'")
        tokenizer.visualize_tokenization(text)
        print()

    # Show debug mode
    print("Debug mode (step-by-step tokenization):")
    tokenizer.encode("Hello world!", debug=True)
    print()


def demo_bpe_tokenizer():
    """Demonstrate BPETokenizer usage"""
    print("=" * 60)
    print("BPE TOKENIZER DEMO")
    print("=" * 60)
    print("BPE (Byte-Pair Encoding) learns vocabulary from data.")
    print("It's the algorithm used by GPT models!")
    print()

    # Create and train BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=100)

    # Training corpus
    texts = [
        "hello world",
        "hello there",
        "world peace",
        "peace and love",
        "hello beautiful world",
        "there is peace in the world",
    ]

    print("Training corpus:")
    for text in texts:
        print(f"  '{text}'")
    print()

    print("Training BPE tokenizer...")
    tokenizer.train(texts, verbose=True)
    print()

    # Show what was learned
    print(f"Learned vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Number of merges: {len(tokenizer.get_merges())}")
    print()

    # Test encoding/decoding
    test_texts = [
        "hello world",
        "hello beautiful world",
        "there is peace",
        "unknown words here",  # Test unknown handling
    ]

    for text in test_texts:
        print(f"Text: '{text}'")
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        print(f"Tokens: {len(encoded)}")
        print()


def demo_comparison():
    """Compare different text types"""
    print("=" * 60)
    print("TOKENIZATION COMPARISON")
    print("=" * 60)
    print("Let's see how different texts get tokenized...")
    print()

    # Create tokenizers
    simple_vocab = load_sample_vocab("simple")
    simple_tokenizer = SimpleTokenizer(simple_vocab)

    bpe_tokenizer = BPETokenizer(vocab_size=100)
    bpe_texts = ["hello world", "hello there", "world peace", "hello beautiful world"]
    bpe_tokenizer.train(bpe_texts, verbose=False)

    # Test different text types
    test_texts = [
        "Hello world!",
        "strawberry",  # The famous example
        "ChatGPT is amazing!",
        "The quick brown fox jumps over the lazy dog.",
        "def hello_world(): print('Hello, world!')",  # Code
        "🚀 AI is the future! 🌟",  # With emojis
    ]

    for text in test_texts:
        print(f"Text: '{text}'")
        print("-" * 40)

        # Simple tokenizer
        simple_tokens = simple_tokenizer.encode(text)
        print(f"SimpleTokenizer: {simple_tokens} ({len(simple_tokens)} tokens)")

        # BPE tokenizer
        bpe_tokens = bpe_tokenizer.encode(text)
        print(f"BPETokenizer:    {bpe_tokens} ({len(bpe_tokens)} tokens)")

        # Show efficiency
        simple_eff = len(text) / len(simple_tokens) if simple_tokens else 0
        bpe_eff = len(text) / len(bpe_tokens) if bpe_tokens else 0
        print(f"Efficiency:      {simple_eff:.1f} vs {bpe_eff:.1f} chars/token")
        print()


def demo_strawberry_problem():
    """Demonstrate the famous strawberry problem"""
    print("=" * 60)
    print("THE STRAWBERRY PROBLEM")
    print("=" * 60)
    print("Why GPT can't count letters reliably...")
    print()

    # Create tokenizer that might split "strawberry" in different ways
    vocab = {
        "straw": 1,
        "berry": 2,
        "strawberry": 3,  # Full word
        "strawberries": 4,  # Plural
        " ": 5,
        "s": 6,
    }

    tokenizer = SimpleTokenizer(vocab)
    demonstrate_strawberry_problem(tokenizer)


def demo_advanced_features():
    """Demonstrate advanced tokenizer features"""
    print("=" * 60)
    print("ADVANCED FEATURES")
    print("=" * 60)

    # Create tokenizer
    vocab = load_sample_vocab("english")
    tokenizer = SimpleTokenizer(vocab)

    text = "Hello world! How are you today?"

    print("1. Token visualization:")
    visualize_tokens(text, tokenizer)
    print()

    print("2. Efficiency analysis:")
    from mini_gpt.utils import analyze_text_efficiency

    efficiency = analyze_text_efficiency(text, tokenizer)
    for key, value in efficiency.items():
        print(f"   {key}: {value:.3f}")
    print()

    print("3. Statistics:")
    stats = tokenizer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()

    print("4. Save and load:")
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        tokenizer.save(temp_path)
        print(f"   Saved to: {temp_path}")

        # Load it back
        new_tokenizer = SimpleTokenizer()
        new_tokenizer.load(temp_path)

        # Test it works
        result = new_tokenizer.encode("Hello world!")
        print(f"   Loaded tokenizer result: {result}")

    finally:
        os.unlink(temp_path)

    print()


def main():
    """Run all demonstrations"""
    print("🚀 MiniGPT Tokenizer Demo")
    print("=" * 60)
    print("Welcome to Part 1: How GPT Reads Your Words!")
    print("This demo shows the magic behind tokenization.")
    print()

    try:
        demo_simple_tokenizer()
        demo_bpe_tokenizer()
        demo_comparison()
        demo_strawberry_problem()
        demo_advanced_features()

        print("=" * 60)
        print("🎉 Demo complete!")
        print()
        print("Key takeaways:")
        print("• Tokenization converts text to numbers")
        print("• Different algorithms produce different results")
        print("• BPE learns vocabulary from data")
        print("• Token boundaries affect model capabilities")
        print()
        print("Next steps:")
        print("• Read the Part 1 blog post")
        print("• Try the Jupyter notebook")
        print("• Experiment with your own texts!")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure you've installed the package: pip install -e .")


if __name__ == "__main__":
    main()

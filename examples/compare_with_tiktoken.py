#!/usr/bin/env python3
"""
Compare MiniGPT tokenizers with OpenAI's tiktoken

Shows the difference between our educational tokenizers
and production tokenizers like tiktoken.

Requires: pip install tiktoken
"""

from mini_gpt.tokenizer import BPETokenizer, SimpleTokenizer
from mini_gpt.utils import load_sample_vocab


def compare_tokenizers():
    """Compare all three tokenizers on same text"""

    print("=" * 80)
    print("TOKENIZER COMPARISON: MiniGPT vs tiktoken")
    print("=" * 80)
    print("Comparing our educational tokenizers with OpenAI's production tokenizer")
    print()

    # Create our tokenizers
    simple_vocab = load_sample_vocab("english")
    simple_tokenizer = SimpleTokenizer(simple_vocab)

    bpe_tokenizer = BPETokenizer(vocab_size=200)
    training_texts = [
        "hello world",
        "hello there",
        "world peace",
        "peace and love",
        "the quick brown fox",
        "jumps over the lazy dog",
        "hello beautiful world",
    ]
    bpe_tokenizer.train(training_texts, verbose=False)

    # Test examples
    examples = [
        "Hello world!",
        "strawberry",
        "ChatGPT is amazing!",
        "The quick brown fox jumps over the lazy dog.",
        "def hello_world(): print('Hello, world!')",  # Code
        "🚀 AI is the future! 🌟",  # With emojis
    ]

    for text in examples:
        print(f"Text: '{text}'")
        print("-" * 60)

        # Our SimpleTokenizer
        simple_tokens = simple_tokenizer.encode(text)
        print(f"SimpleTokenizer: {simple_tokens} ({len(simple_tokens)} tokens)")

        # Our BPETokenizer
        bpe_tokens = bpe_tokenizer.encode(text)
        print(f"BPETokenizer:    {bpe_tokens} ({len(bpe_tokens)} tokens)")

        # tiktoken (if available)
        try:
            import tiktoken

            tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")
            tiktoken_tokens = tiktoken_tokenizer.encode(text)
            print(f"tiktoken:        {tiktoken_tokens} ({len(tiktoken_tokens)} tokens)")

            # Show actual tokens
            tiktoken_token_strings = tiktoken_tokenizer.decode_tokens_bytes(tiktoken_tokens)
            print(
                f"tiktoken tokens: {[t.decode('utf-8', errors='replace') for t in tiktoken_token_strings]}"
            )

        except ImportError:
            print("tiktoken:        Not available (install with: pip install tiktoken)")
        except Exception as e:
            print(f"tiktoken:        Error - {e}")

        # Efficiency comparison
        simple_eff = len(text) / len(simple_tokens) if simple_tokens else 0
        bpe_eff = len(text) / len(bpe_tokens) if bpe_tokens else 0

        print(f"Efficiency:      {simple_eff:.1f} vs {bpe_eff:.1f} chars/token", end="")

        try:
            tiktoken_eff = len(text) / len(tiktoken_tokens) if tiktoken_tokens else 0
            print(f" vs {tiktoken_eff:.1f} chars/token")
        except Exception:
            print()

        print()


def show_tiktoken_quirks():
    """Demonstrate interesting tiktoken behaviors"""

    print("=" * 80)
    print("TIKTOKEN QUIRKS AND BEHAVIORS")
    print("=" * 80)
    print("Production tokenizers have interesting behaviors...")
    print()

    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Test cases that show interesting behaviors
        test_cases = [
            ("Leading spaces", " hello world"),
            ("Multiple spaces", "hello    world"),
            ("Special characters", "Hello, world! @#$%"),
            ("Unicode", "Hello 世界! 🌟"),
            ("Code", "def hello(): return 'world'"),
            ("Numbers", "123 456.789"),
            ("Mixed case", "Hello WORLD hello"),
        ]

        for description, text in test_cases:
            print(f"{description}: '{text}'")
            tokens = tokenizer.encode(text)
            token_strings = [
                t.decode("utf-8", errors="replace") for t in tokenizer.decode_tokens_bytes(tokens)
            ]

            print(f"  Tokens: {tokens}")
            print(f"  Strings: {token_strings}")
            print(f"  Count: {len(tokens)}")
            print()

        # Show vocabulary size
        print(f"tiktoken vocabulary size: {tokenizer.n_vocab}")
        print()

        # Show some interesting tokens
        print("Some interesting tiktoken tokens:")
        interesting_ids = [0, 1, 2, 3, 100, 1000, 10000, 50000, 100000]
        for token_id in interesting_ids:
            if token_id < tokenizer.n_vocab:
                try:
                    token_bytes = tokenizer.decode_single_token_bytes(token_id)
                    token_str = token_bytes.decode("utf-8", errors="replace")
                    print(f"  {token_id}: '{token_str}'")
                except Exception:
                    print(f"  {token_id}: <error>")

    except ImportError:
        print("tiktoken not available. Install with: pip install tiktoken")
    except Exception as e:
        print(f"Error with tiktoken: {e}")


def demonstrate_educational_vs_production():
    """Show the differences between educational and production tokenizers"""

    print("=" * 80)
    print("EDUCATIONAL vs PRODUCTION TOKENIZERS")
    print("=" * 80)
    print()

    print("Our educational tokenizers:")
    print("✅ Simple and easy to understand")
    print("✅ Well-commented code")
    print("✅ Clear algorithm implementation")
    print("✅ Good for learning concepts")
    print("❌ Not optimized for performance")
    print("❌ Limited vocabulary")
    print("❌ Basic error handling")
    print()

    print("Production tokenizers (like tiktoken):")
    print("✅ Highly optimized for speed")
    print("✅ Large, carefully curated vocabularies")
    print("✅ Robust error handling")
    print("✅ Handles edge cases well")
    print("❌ Complex implementation")
    print("❌ Hard to understand internals")
    print("❌ Optimized for specific use cases")
    print()

    print("Why we built educational versions:")
    print("• To understand the core concepts")
    print("• To see how algorithms work step-by-step")
    print("• To experiment with different approaches")
    print("• To build intuition before using production tools")
    print()


def main():
    """Run the comparison demo"""
    print("🔍 MiniGPT vs tiktoken Comparison")
    print("=" * 80)
    print("Comparing our educational tokenizers with OpenAI's production tokenizer")
    print()

    try:
        compare_tokenizers()
        show_tiktoken_quirks()
        demonstrate_educational_vs_production()

        print("=" * 80)
        print("🎉 Comparison complete!")
        print()
        print("Key insights:")
        print("• Production tokenizers are highly optimized")
        print("• Educational tokenizers help you understand the concepts")
        print("• Both approaches have their place in learning")
        print("• Understanding the basics helps you use production tools better")
        print()
        print("Next steps:")
        print("• Try different text types with our tokenizers")
        print("• Experiment with different vocabulary sizes")
        print("• Read about BPE algorithm in detail")
        print("• Move on to Part 2: Embeddings!")

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        print("Make sure you've installed the package: pip install -e .")
        print("For tiktoken comparison, install: pip install tiktoken")


if __name__ == "__main__":
    main()

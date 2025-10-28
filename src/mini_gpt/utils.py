"""
Utility functions for MiniGPT Part 1: Tokenization

This module provides utility functions for working with tokenizers,
visualizing tokenization, and analyzing text efficiency.

Additional utilities for training, embeddings, and model architecture
will be added in later parts of the series.
"""

from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# TOKENIZATION UTILITIES
# ============================================================================


def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in text.

    Args:
        text: Input text
        tokenizer: Tokenizer instance with encode() method

    Returns:
        Number of tokens

    Example:
        >>> tokenizer = SimpleTokenizer(vocab)
        >>> count_tokens("Hello world!", tokenizer)
        3
    """
    return len(tokenizer.encode(text))


def visualize_tokens(text: str, tokenizer, max_length: int = 50, show_ids: bool = True) -> None:
    """
    Pretty-print how text gets tokenized.

    Shows original text, tokens, and IDs in a readable format.
    Makes spaces and special characters visible with symbols.

    Args:
        text: Input text to visualize
        tokenizer: Tokenizer instance
        max_length: Maximum text length to display
        show_ids: Whether to show token IDs

    Example:
        >>> visualize_tokens("Hello world!", tokenizer)
        ============================================================
        TOKENIZATION VISUALIZATION
        ============================================================
        Original: "Hello world!"
        Tokens:   ['Hello', '␣world', '!']
        IDs:      [102, 103, 104]
        Count:    3 tokens
    """
    # Truncate if too long
    display_text = text[:max_length] + "..." if len(text) > max_length else text

    # Get tokens
    token_ids = tokenizer.encode(display_text)
    token_strings = []

    # Convert IDs to token strings
    for tid in token_ids:
        if hasattr(tokenizer, "id_to_token"):
            token_str = tokenizer.id_to_token(tid)
            # Make spaces visible
            token_str = token_str.replace(" ", "␣")
            token_str = token_str.replace("\n", "↵")
            token_str = token_str.replace("\t", "⇥")
            token_strings.append(token_str)
        else:
            token_strings.append(f"ID:{tid}")

    # Print visualization
    print("=" * 60)
    print("TOKENIZATION VISUALIZATION")
    print("=" * 60)
    print(f'Original: "{display_text}"')
    print(f"Tokens:   {token_strings}")

    if show_ids:
        print(f"IDs:      {token_ids}")

    print(f"Count:    {len(token_ids)} token{'s' if len(token_ids) != 1 else ''}")
    print("=" * 60)


def compare_tokenizers(text: str, tokenizers: List, names: Optional[List[str]] = None) -> None:
    """
    Compare how different tokenizers handle the same text.

    Useful for understanding differences between tokenization strategies
    (e.g., character-level vs BPE vs WordPiece).

    Args:
        text: Input text to compare
        tokenizers: List of tokenizer instances
        names: Optional list of names for each tokenizer

    Example:
        >>> compare_tokenizers(
        ...     "Hello world!",
        ...     [simple_tok, bpe_tok],
        ...     ["Simple", "BPE"]
        ... )
    """
    if names is None:
        names = [f"Tokenizer {i + 1}" for i in range(len(tokenizers))]

    print("=" * 80)
    print("TOKENIZER COMPARISON")
    print("=" * 80)
    print(f'Text: "{text}"')
    print()

    for tokenizer, name in zip(tokenizers, names):
        try:
            token_ids = tokenizer.encode(text)

            # Get token strings if possible
            if hasattr(tokenizer, "id_to_token"):
                token_strings = [tokenizer.id_to_token(tid) for tid in token_ids]
                # Make spaces visible
                token_strings = [t.replace(" ", "␣") for t in token_strings]
            else:
                token_strings = [f"ID:{tid}" for tid in token_ids]

            print(f"{name}:")
            print(f"  Tokens: {token_strings}")
            print(f"  Count:  {len(token_ids)} token{'s' if len(token_ids) != 1 else ''}")
            print()

        except Exception as e:
            print(f"{name}: ERROR - {str(e)}")
            print()


def analyze_text_efficiency(text: str, tokenizer) -> Dict[str, float]:
    """
    Analyze tokenization efficiency metrics.

    Calculates how efficiently the tokenizer compresses text,
    measured by characters per token.

    Args:
        text: Input text to analyze
        tokenizer: Tokenizer instance

    Returns:
        Dictionary with efficiency metrics:
        - char_count: Number of characters
        - token_count: Number of tokens
        - chars_per_token: Average characters per token
        - compression_ratio: Same as chars_per_token
        - efficiency_score: Inverse of chars_per_token

    Example:
        >>> metrics = analyze_text_efficiency("Hello world!", tokenizer)
        >>> print(f"Efficiency: {metrics['chars_per_token']:.2f} chars/token")
    """
    char_count = len(text)
    token_ids = tokenizer.encode(text)
    token_count = len(token_ids)

    chars_per_token = char_count / token_count if token_count > 0 else 0

    return {
        "char_count": char_count,
        "token_count": token_count,
        "chars_per_token": chars_per_token,
        "compression_ratio": chars_per_token,
        "efficiency_score": 1.0 / chars_per_token if chars_per_token > 0 else 0,
    }


def create_tokenization_report(text: str, tokenizer, title: str = "Tokenization Report") -> None:
    """
    Create a comprehensive tokenization report.

    Displays detailed analysis of how text is tokenized,
    including token breakdown and efficiency metrics.

    Args:
        text: Input text to analyze
        tokenizer: Tokenizer instance
        title: Report title
    """
    print("=" * 60)
    print(title.upper())
    print("=" * 60)

    # Basic stats
    char_count = len(text)
    token_ids = tokenizer.encode(text)
    token_count = len(token_ids)

    # Truncate long text for display
    display_text = text[:100] + "..." if len(text) > 100 else text

    print(f'Text: "{display_text}"')
    print(f"Character count: {char_count}")
    print(f"Token count: {token_count}")
    print(
        f"Compression ratio: {char_count / token_count if token_count > 0 else 0:.2f} chars/token"
    )
    print()

    # Token breakdown
    if hasattr(tokenizer, "id_to_token"):
        token_strings = [tokenizer.id_to_token(tid) for tid in token_ids]
        print("Token breakdown:")

        # Show first 20 tokens if many
        display_count = min(20, len(token_strings))
        for i in range(display_count):
            token = token_strings[i].replace(" ", "␣").replace("\n", "↵")
            print(f"  {i + 1:2d}. '{token}' (ID: {token_ids[i]})")

        if len(token_strings) > display_count:
            print(f"  ... and {len(token_strings) - display_count} more tokens")
    else:
        print("Token IDs:", token_ids[:20])
        if len(token_ids) > 20:
            print(f"... and {len(token_ids) - 20} more tokens")

    print()

    # Efficiency analysis
    efficiency = analyze_text_efficiency(text, tokenizer)
    print("Efficiency metrics:")
    for key, value in efficiency.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("=" * 60)


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================


def demonstrate_strawberry_problem(tokenizer) -> None:
    """
    Demonstrate the famous 'strawberry problem' with tokenization.

    Shows why GPT models struggle with character-level tasks like
    counting letters. The problem: tokenization breaks words into
    subword units, losing the character-level structure.

    Args:
        tokenizer: Tokenizer instance to demonstrate with

    Note:
        This demo is most effective with tokenizers that split
        'strawberry' into subwords like ['straw', 'berry'].
    """
    print("=" * 60)
    print("THE STRAWBERRY PROBLEM")
    print("=" * 60)
    print("Why GPT can't count letters reliably...")
    print()
    print("When you ask GPT 'How many R's are in strawberry?',")
    print("it doesn't see individual letters. It sees tokens!")
    print()

    # Test the word "strawberry"
    test_word = "strawberry"
    token_ids = tokenizer.encode(test_word)

    # Get token strings if possible
    if hasattr(tokenizer, "id_to_token"):
        token_strings = [tokenizer.id_to_token(tid) for tid in token_ids]

        print("You see:  s-t-r-a-w-b-e-r-r-y (10 letters, 3 R's)")
        print(
            f"GPT sees: {token_strings} ({len(token_strings)} token{'s' if len(token_strings) != 1 else ''})"
        )
        print()
        print("GPT doesn't have direct access to the letters!")
        print("It only knows about tokens: " + str(token_strings))
    else:
        print(f"Text: '{test_word}'")
        print(f"Tokens: {len(token_ids)} token(s)")
        print(f"Token IDs: {token_ids}")

    print()
    print("This is why GPT models:")
    print("  ❌ Struggle to count letters")
    print("  ❌ Can't reliably spell backwards")
    print("  ❌ Have difficulty with character-level tasks")
    print()
    print("Modern models (like GPT-4+) learned to work around this")
    print("through better reasoning, but tokenization still happens!")
    print("=" * 60)


# ============================================================================
# VOCABULARY UTILITIES
# ============================================================================


def load_sample_vocab(style: str = "simple") -> Dict[str, int]:
    """
    Load a sample vocabulary for testing and demonstrations.

    Creates vocabularies with continuous token IDs (0, 1, 2, ...).
    Special tokens (<UNK>, <PAD>, etc.) always have IDs 0-3.

    Args:
        style: Vocabulary style - "simple", "english", or "code"

    Returns:
        Dictionary mapping tokens to continuous IDs

    Raises:
        ValueError: If style is not recognized

    Example:
        >>> vocab = load_sample_vocab("simple")
        >>> vocab["<UNK>"]  # Unknown token
        0
        >>> vocab["straw"]  # Regular token
        12
    """
    if style == "simple":
        # Define tokens in order (special tokens first)
        tokens = [
            # Special tokens (IDs 0-3)
            "<UNK>",
            "<PAD>",
            "<BOS>",
            "<EOS>",
            # Common words (IDs 4+)
            "the",
            "a",
            "Hello",
            " world",
            "!",
            "?",
            ".",
            ",",
            " ",
            # Subwords for demonstration (IDs 13+)
            "straw",
            "berry",
            "ing",
            "ed",
            "er",
            "s",
            "ly",
        ]

    elif style == "english":
        tokens = [
            # Special tokens
            "<UNK>",
            "<PAD>",
            "<BOS>",
            "<EOS>",
            # Common English words (from frequency lists)
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "I",
            "it",
            "for",
            "not",
            "on",
            "with",
            "he",
            "as",
            "you",
            "do",
            "at",
            "this",
            "but",
            "his",
            "by",
            "from",
            "they",
            "we",
            "say",
            "her",
            "she",
            # Common punctuation
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            '"',
            "'",
            "(",
            ")",
            " ",
            "\n",
        ]

    elif style == "code":
        tokens = [
            # Special tokens
            "<UNK>",
            "<PAD>",
            "<BOS>",
            "<EOS>",
            # Python keywords
            "def",
            "class",
            "if",
            "else",
            "for",
            "while",
            "return",
            "import",
            "from",
            "as",
            "try",
            "except",
            "finally",
            "with",
            "lambda",
            "True",
            "False",
            "None",
            "and",
            "or",
            "not",
            "in",
            "is",
            # Operators
            "=",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "+",
            "-",
            "*",
            "/",
            "%",
            "//",
            # Brackets and punctuation
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            ":",
            ",",
            ".",
            " ",
            "\n",
            "    ",
        ]

    else:
        raise ValueError(
            f"Unknown vocabulary style: '{style}'. Choose from: 'simple', 'english', or 'code'"
        )

    # Create vocab with continuous IDs (0, 1, 2, 3, ...)
    return {token: idx for idx, token in enumerate(tokens)}


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================


def load_text_data(filepath: str) -> str:
    """
    Load text data from a file.

    Args:
        filepath: Path to the text file

    Returns:
        Loaded text content

    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file encoding is not UTF-8

    Example:
        >>> text = load_text_data("data/tiny_shakespeare.txt")
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error. Expected UTF-8: {path}\nError: {str(e)}") from e


def save_text_data(text: str, filepath: str) -> None:
    """
    Save text data to a file.

    Creates parent directories if they don't exist.

    Args:
        text: Text content to save
        filepath: Path where to save the file

    Example:
        >>> save_text_data("Hello world!", "output/test.txt")
    """
    path = Path(filepath)

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def format_number(num: int) -> str:
    """
    Format a number with appropriate units (K, M, B).

    Args:
        num: Number to format

    Returns:
        Formatted string with unit suffix

    Example:
        >>> format_number(1500)
        '1.5K'
        >>> format_number(2500000)
        '2.5M'
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


# ============================================================================
# TODO: Additional utilities will be added in future parts
# ============================================================================
# Part 2: Embedding utilities
# Part 3: Attention visualization utilities
# Part 4: Training utilities (plot_training_curves, etc.)
# Part 5: Generation utilities

"""
Tokenizer implementation for MiniGPT
Educational implementations of tokenization techniques

This module provides two tokenizer implementations:
1. SimpleTokenizer: Dictionary-based, demonstrates core concepts
2. BPETokenizer: Byte-Pair Encoding, more realistic implementation

Both are designed for educational purposes with clear, well-commented code.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# Special tokens used across all tokenizers
UNK_TOKEN = "<UNK>"
UNK_ID = 0
PAD_TOKEN = "<PAD>"
PAD_ID = 1
BOS_TOKEN = "<BOS>"
BOS_ID = 2
EOS_TOKEN = "<EOS>"
EOS_ID = 3

# Training parameters
DEFAULT_VOCAB_SIZE = 1000
MIN_VOCAB_SIZE = 100
MAX_VOCAB_SIZE = 50000


class SimpleTokenizer:
    """
    A simple dictionary-based tokenizer for educational purposes.

    This tokenizer demonstrates the core idea of tokenization:
    breaking text into known chunks (tokens) and mapping them to IDs.

    Algorithm: Greedy longest-match from left to right

    Example:
        >>> vocab = {"Hello": 1, " world": 2, "!": 3}
        >>> tokenizer = SimpleTokenizer(vocab)
        >>> tokenizer.encode("Hello world!")
        [1, 2, 3]
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize the tokenizer with a vocabulary.

        Args:
            vocab: Dictionary mapping tokens to IDs. If None, creates empty vocab.
        """
        self.vocab = vocab or {}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Ensure special tokens exist
        if UNK_TOKEN not in self.vocab:
            self.vocab[UNK_TOKEN] = UNK_ID
            self.reverse_vocab[UNK_ID] = UNK_TOKEN

    def encode(self, text: str, debug: bool = False) -> List[int]:
        """
        Convert text to token IDs using greedy longest-match algorithm.

        Uses greedy longest-match algorithm: tries to match the longest
        possible substring from the vocabulary, working left to right.

        Args:
            text: Input text to tokenize
            debug: If True, print step-by-step tokenization

        Returns:
            List of integer token IDs

        Example:
            >>> tokenizer.encode("Hello world!")
            [1, 2, 3]

        Note:
            Unknown characters are replaced with the <UNK> token (ID: 0)
        """
        if not text:
            return []

        if debug:
            print(f"Tokenizing: '{text}'")

        tokens = []
        i = 0

        while i < len(text):
            # Try to find the longest match starting at position i
            longest_match = ""
            longest_match_id = UNK_ID

            # Check all possible substrings starting at position i
            for j in range(i + 1, len(text) + 1):
                substring = text[i:j]
                if substring in self.vocab:
                    longest_match = substring
                    longest_match_id = self.vocab[substring]

            if longest_match:
                # Found a match, add it to tokens
                tokens.append(longest_match_id)
                if debug:
                    print(f"  Matched '{longest_match}' -> {longest_match_id}")
                i += len(longest_match)
            else:
                # No match found, use UNK token
                tokens.append(UNK_ID)
                if debug:
                    print(f"  Unknown char '{text[i]}' -> {UNK_ID}")
                i += 1

        if debug:
            print(f"Result: {tokens}")

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string

        Example:
            >>> tokenizer.decode([1, 2, 3])
            "Hello world!"
        """
        if not token_ids:
            return ""

        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                # Skip special tokens when decoding
                if token not in [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                    tokens.append(token)
            else:
                # Unknown token ID, use UNK character
                tokens.append("?")

        return "".join(tokens)

    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.

        Returns:
            Number of tokens in vocabulary
        """
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        """
        Get ID for a specific token.

        Args:
            token: Token string

        Returns:
            Token ID, or UNK_ID if token not found
        """
        return self.vocab.get(token, UNK_ID)

    def id_to_token(self, token_id: int) -> str:
        """
        Get token for a specific ID.

        Args:
            token_id: Token ID

        Returns:
            Token string, or UNK_TOKEN if ID not found
        """
        return self.reverse_vocab.get(token_id, UNK_TOKEN)

    def visualize_tokenization(self, text: str) -> None:
        """
        Print a visual representation of how text is tokenized.

        Args:
            text: Input text to visualize

        Example output:
            Input:  "Hello world!"
            Tokens: ["Hello", " world", "!"]
            IDs:    [102, 103, 301]
        """
        tokens = self.encode(text)
        token_strings = [self.id_to_token(tid) for tid in tokens]

        print(f'Input:  "{text}"')
        print(f"Tokens: {token_strings}")
        print(f"IDs:    {tokens}")

    def get_stats(self) -> Dict[str, int]:
        """
        Return tokenizer statistics.

        Returns:
            Dictionary with vocab_size and other stats
        """
        return {
            "vocab_size": len(self.vocab),
            "special_tokens": len([t for t in self.vocab.keys() if t.startswith("<")]),
        }

    def save(self, filepath: str) -> None:
        """
        Save tokenizer to file.

        Args:
            filepath: Path to save the tokenizer
        """
        data = {"vocab": self.vocab, "type": "SimpleTokenizer"}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load tokenizer from file.

        Args:
            filepath: Path to load the tokenizer from
        """
        with open(filepath) as f:
            data = json.load(f)

        if data.get("type") != "SimpleTokenizer":
            raise ValueError("File is not a SimpleTokenizer")

        self.vocab = data["vocab"]
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer.

    BPE is the algorithm used by GPT models. It learns a vocabulary
    by iteratively merging the most frequent character pairs.

    Training process:
    1. Start with character-level tokens
    2. Find most frequent pair
    3. Merge that pair into new token
    4. Repeat until vocab_size reached

    Example:
        >>> tokenizer = BPETokenizer(vocab_size=1000)
        >>> tokenizer.train(["hello world", "hello there"])
        >>> tokenizer.encode("hello")
        [42, 108]  # Example IDs
    """

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE):
        """
        Initialize BPE tokenizer with target vocabulary size.

        Args:
            vocab_size: Target vocabulary size (including special tokens)
        """
        if vocab_size < MIN_VOCAB_SIZE or vocab_size > MAX_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be between {MIN_VOCAB_SIZE} and {MAX_VOCAB_SIZE}")

        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []  # List of (token1, token2) pairs that were merged
        self.word_freqs: Counter[str] = Counter()

        # Initialize with special tokens
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        """Add special tokens to vocabulary."""
        special_tokens = [
            (UNK_TOKEN, UNK_ID),
            (PAD_TOKEN, PAD_ID),
            (BOS_TOKEN, BOS_ID),
            (EOS_TOKEN, EOS_ID),
        ]

        for token, token_id in special_tokens:
            self.vocab[token] = token_id
            self.reverse_vocab[token_id] = token

    def train(self, texts: List[str], verbose: bool = False) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Args:
            texts: List of training texts
            verbose: Print training progress
        """
        if not texts:
            raise ValueError("Cannot train on empty corpus")

        if verbose:
            print(f"Training BPE tokenizer on {len(texts)} texts...")
            print(f"Target vocabulary size: {self.vocab_size}")

        # Step 1: Count word frequencies
        self._count_word_frequencies(texts)

        if verbose:
            print(f"Found {len(self.word_freqs)} unique words")

        # Step 2: Initialize with character-level tokens
        self._initialize_character_tokens()

        if verbose:
            print(f"Character-level vocab size: {len(self.vocab)}")

        # Step 3: Iteratively merge most frequent pairs
        while len(self.vocab) < self.vocab_size:
            # Find most frequent pair
            pair_freq = self._count_pairs()
            if not pair_freq:
                break  # No more pairs to merge

            # Get most frequent pair
            most_frequent_pair = max(pair_freq, key=lambda x: pair_freq[x])

            # Merge the pair
            self._merge_pair(most_frequent_pair)
            self.merges.append(most_frequent_pair)

            if verbose and len(self.vocab) % 100 == 0:
                print(f"Vocab size: {len(self.vocab)}")

        if verbose:
            print(f"Training complete! Final vocab size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")

    def _count_word_frequencies(self, texts: List[str]) -> None:
        """Count frequency of each word in the corpus."""
        self.word_freqs = Counter()

        for text in texts:
            # Simple word splitting (can be improved with proper tokenization)
            words = text.split()
            for word in words:
                # Add special markers for BPE
                self.word_freqs[f"▁{word}"] += 1  # ▁ indicates word boundary

    def _initialize_character_tokens(self) -> None:
        """Initialize vocabulary with all unique characters."""
        # Get all unique characters from the corpus
        chars: set[str] = set()
        for word in self.word_freqs.keys():
            chars.update(word)

        # Add characters to vocabulary
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
                self.reverse_vocab[len(self.vocab) - 1] = char

    def _count_pairs(self) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent token pairs."""
        pair_freq: Dict[Tuple[str, str], int] = defaultdict(int)

        for word, freq in self.word_freqs.items():
            # Split word into current tokens
            tokens = self._word_to_tokens(word)

            # Count adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freq[pair] += freq

        return dict(pair_freq)

    def _word_to_tokens(self, word: str) -> List[str]:
        """Convert word to list of current tokens."""
        # Start with character-level tokens
        tokens = list(word)

        # Apply all learned merges
        for token1, token2 in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == token1 and tokens[i + 1] == token2:
                    # Found the pair, merge it
                    new_tokens.append(token1 + token2)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def _merge_pair(self, pair: Tuple[str, str]) -> None:
        """Merge a pair of tokens into a new token."""
        token1, token2 = pair
        merged_token = token1 + token2

        # Add merged token to vocabulary
        self.vocab[merged_token] = len(self.vocab)
        self.reverse_vocab[len(self.vocab) - 1] = merged_token

    def encode(self, text: str) -> List[int]:
        """
        Encode text using learned BPE merges.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        if not text:
            return []

        # Split into words and add word boundary markers
        words = text.split()
        token_ids = []

        for word in words:
            # Add word boundary marker
            word_with_marker = f"▁{word}"

            # Convert to tokens using learned merges
            tokens = self._word_to_tokens(word_with_marker)

            # Convert tokens to IDs
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(UNK_ID)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        if not token_ids:
            return ""

        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                # Skip special tokens
                if token not in [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                    tokens.append(token)
            else:
                tokens.append("?")  # Unknown token ID

        # Join tokens and clean up word boundaries
        text = "".join(tokens)
        text = text.replace("▁", " ")  # Replace word boundary markers with spaces
        text = text.strip()  # Remove leading/trailing spaces

        return text

    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return the learned vocabulary."""
        return self.vocab.copy()

    def get_merges(self) -> List[Tuple[str, str]]:
        """Return the list of learned merges."""
        return self.merges.copy()

    def save(self, path: str) -> None:
        """
        Save trained tokenizer to file.

        Args:
            path: Path to save the tokenizer
        """
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
            "type": "BPETokenizer",
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load trained tokenizer from file.

        Args:
            path: Path to load the tokenizer from
        """
        with open(path) as f:
            data = json.load(f)

        if data.get("type") != "BPETokenizer":
            raise ValueError("File is not a BPETokenizer")

        self.vocab = data["vocab"]
        self.merges = [tuple(merge) for merge in data["merges"]]
        self.vocab_size = data["vocab_size"]
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def get_stats(self) -> Dict[str, int]:
        """
        Return tokenizer statistics.

        Returns:
            Dictionary with various statistics
        """
        return {
            "vocab_size": len(self.vocab),
            "merges_count": len(self.merges),
            "special_tokens": len([t for t in self.vocab.keys() if t.startswith("<")]),
            "target_vocab_size": self.vocab_size,
        }


# Legacy CharacterTokenizer for backward compatibility
class CharacterTokenizer:
    """
    Simple character-level tokenizer for educational purposes.

    This tokenizer converts text to individual characters, which is
    simpler to understand but less efficient than BPE.
    """

    def __init__(self):
        """Initialize the character tokenizer."""
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build vocabulary from common characters."""
        # Add special tokens
        special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

        # Add printable ASCII characters
        chars = [chr(i) for i in range(32, 127)]

        # Build vocabulary
        all_tokens = special_tokens + chars
        self.vocab = {token: i for i, token in enumerate(all_tokens)}
        self.reverse_vocab = {i: token for token, i in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Convert text to character token IDs.

        Args:
            text: Input text to tokenize

        Returns:
            List of character token IDs
        """
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab[UNK_TOKEN])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        chars = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                char = self.reverse_vocab[token_id]
                if char not in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                    chars.append(char)
        return "".join(chars)

    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.

        Returns:
            Number of tokens in vocabulary
        """
        return len(self.vocab)

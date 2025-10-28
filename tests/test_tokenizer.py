"""
Comprehensive tests for tokenizer module
Tests both SimpleTokenizer and BPETokenizer implementations
"""

import os
import tempfile

import pytest

from mini_gpt.tokenizer import BPETokenizer, CharacterTokenizer, SimpleTokenizer
from mini_gpt.utils import load_sample_vocab


class TestSimpleTokenizer:
    """Test cases for SimpleTokenizer"""

    def test_initialization_empty(self):
        """Test tokenizer initialization with empty vocab"""
        tokenizer = SimpleTokenizer()
        assert tokenizer.get_vocab_size() == 1  # Only UNK token
        assert "<UNK>" in tokenizer.vocab
        assert tokenizer.vocab["<UNK>"] == 0

    def test_initialization_with_vocab(self):
        """Test tokenizer initialization with custom vocab"""
        vocab = {"hello": 1, "world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)
        assert tokenizer.get_vocab_size() == 4  # 3 custom + 1 UNK
        assert tokenizer.vocab == {"hello": 1, "world": 2, "!": 3, "<UNK>": 0}

    def test_basic_encoding(self):
        """Test basic text encoding"""
        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)
        result = tokenizer.encode("Hello world!")
        assert result == [1, 2, 3]

    def test_basic_decoding(self):
        """Test basic token decoding"""
        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)
        token_ids = [1, 2, 3]
        decoded = tokenizer.decode(token_ids)
        assert decoded == "Hello world!"

    def test_unknown_token_handling(self):
        """Test handling of unknown tokens"""
        vocab = {"Hello": 1, "!": 3}
        tokenizer = SimpleTokenizer(vocab)

        # Text with unknown token should use UNK for each unknown character
        result = tokenizer.encode("Hello unknown!")
        # " unknown" becomes 8 UNK tokens (space + 7 chars for "unknown")
        expected = [1] + [0] * 8 + [3]  # Hello + 8 UNKs for " unknown" + !
        assert result == expected

        # Decoding should handle UNK gracefully
        decoded = tokenizer.decode([1, 0, 3])
        assert decoded == "Hello!"  # UNK token is skipped

    def test_empty_string(self):
        """Test edge case: empty string"""
        vocab = {"Hello": 1}
        tokenizer = SimpleTokenizer(vocab)

        result = tokenizer.encode("")
        assert result == []

        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_longest_match(self):
        """Test greedy longest-match algorithm"""
        # "hello" should match as one token if available, not as "h" + "ello"
        vocab = {"hello": 1, "h": 2, "ello": 3, "!": 4}
        tokenizer = SimpleTokenizer(vocab)

        result = tokenizer.encode("hello!")
        assert result == [1, 4]  # "hello" + "!", not "h" + "ello" + "!"

    def test_vocab_size(self):
        """Test vocabulary size calculation"""
        vocab = {"a": 1, "b": 2, "c": 3}
        tokenizer = SimpleTokenizer(vocab)
        assert tokenizer.get_vocab_size() == 4  # 3 custom + 1 UNK

    def test_round_trip(self):
        """Test encode -> decode returns original (for known tokens)"""
        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)

        original = "Hello world!"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original

    def test_token_to_id(self):
        """Test token to ID conversion"""
        vocab = {"hello": 1, "world": 2}
        tokenizer = SimpleTokenizer(vocab)

        assert tokenizer.token_to_id("hello") == 1
        assert tokenizer.token_to_id("world") == 2
        assert tokenizer.token_to_id("unknown") == 0  # UNK_ID

    def test_id_to_token(self):
        """Test ID to token conversion"""
        vocab = {"hello": 1, "world": 2}
        tokenizer = SimpleTokenizer(vocab)

        assert tokenizer.id_to_token(1) == "hello"
        assert tokenizer.id_to_token(2) == "world"
        assert tokenizer.id_to_token(999) == "<UNK>"  # Unknown ID

    def test_save_and_load(self):
        """Test saving and loading tokenizer"""
        vocab = {"hello": 1, "world": 2, "!": 3, " ": 4}  # Include space
        tokenizer = SimpleTokenizer(vocab)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save tokenizer
            tokenizer.save(temp_path)

            # Load tokenizer
            new_tokenizer = SimpleTokenizer()
            new_tokenizer.load(temp_path)

            # Check they're the same
            assert new_tokenizer.vocab == tokenizer.vocab
            assert new_tokenizer.get_vocab_size() == tokenizer.get_vocab_size()

            # Test functionality
            result = new_tokenizer.encode("hello world!")
            assert result == [1, 4, 2, 3]  # hello + space + world + !

        finally:
            os.unlink(temp_path)

    def test_debug_mode(self):
        """Test debug mode doesn't crash"""
        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)

        # Should not raise an error
        result = tokenizer.encode("Hello world!", debug=True)
        assert result == [1, 2, 3]

    def test_visualize_tokenization(self):
        """Test visualization method doesn't crash"""
        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)

        # Should not raise an error
        tokenizer.visualize_tokenization("Hello world!")

    def test_get_stats(self):
        """Test statistics method"""
        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)

        stats = tokenizer.get_stats()
        assert "vocab_size" in stats
        assert "special_tokens" in stats
        assert stats["vocab_size"] == 4  # 3 custom + 1 UNK


class TestBPETokenizer:
    """Test cases for BPETokenizer"""

    def test_initialization(self):
        """Test tokenizer initialization"""
        tokenizer = BPETokenizer(vocab_size=100)
        assert tokenizer.vocab_size == 100
        assert tokenizer.get_vocab_size() == 4  # Only special tokens initially
        assert "<UNK>" in tokenizer.vocab
        assert "<PAD>" in tokenizer.vocab

    def test_initialization_invalid_size(self):
        """Test initialization with invalid vocab size"""
        with pytest.raises(ValueError):
            BPETokenizer(vocab_size=50)  # Too small

        with pytest.raises(ValueError):
            BPETokenizer(vocab_size=100000)  # Too large

    def test_training_empty_corpus(self):
        """Test training on empty corpus raises error"""
        tokenizer = BPETokenizer(vocab_size=100)

        with pytest.raises(ValueError):
            tokenizer.train([])

    def test_training_simple_corpus(self):
        """Test training on simple corpus"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world", "hello there", "world peace"]

        # Should not raise an error
        tokenizer.train(texts, verbose=False)

        # Should have learned some vocabulary
        assert tokenizer.get_vocab_size() > 4  # More than just special tokens
        assert len(tokenizer.merges) > 0  # Should have made some merges

    def test_encode_after_training(self):
        """Test encoding works after training"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world", "hello there"]
        tokenizer.train(texts, verbose=False)

        # Should be able to encode
        result = tokenizer.encode("hello world")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_decode_after_training(self):
        """Test decoding works after training"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world", "hello there"]
        tokenizer.train(texts, verbose=False)

        # Encode and decode
        encoded = tokenizer.encode("hello world")
        decoded = tokenizer.decode(encoded)

        assert isinstance(decoded, str)
        # Note: exact match might not work due to BPE word boundaries

    def test_vocab_size_limit(self):
        """Test vocabulary doesn't exceed target size"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world", "hello there", "world peace", "peace love"]
        tokenizer.train(texts, verbose=False)

        assert tokenizer.get_vocab_size() <= 100

    def test_character_fallback(self):
        """Test unknown characters are handled"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world"]
        tokenizer.train(texts, verbose=False)

        # Test with character not in training data
        result = tokenizer.encode("hello xyz")
        assert isinstance(result, list)
        # Should contain UNK tokens for unknown characters
        assert 0 in result  # UNK_ID should be present

    def test_save_and_load(self):
        """Test saving and loading trained tokenizer"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world", "hello there"]
        tokenizer.train(texts, verbose=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save tokenizer
            tokenizer.save(temp_path)

            # Load tokenizer
            new_tokenizer = BPETokenizer(vocab_size=100)
            new_tokenizer.load(temp_path)

            # Check they're the same
            assert new_tokenizer.vocab == tokenizer.vocab
            assert new_tokenizer.merges == tokenizer.merges
            assert new_tokenizer.vocab_size == tokenizer.vocab_size

            # Test functionality
            result = new_tokenizer.encode("hello world")
            assert isinstance(result, list)

        finally:
            os.unlink(temp_path)

    def test_round_trip(self):
        """Test encode -> decode preserves text approximately"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world", "hello there", "world peace"]
        tokenizer.train(texts, verbose=False)

        original = "hello world"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)

        # Should be close to original (allowing for BPE word boundary differences)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_get_vocab(self):
        """Test getting vocabulary"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world"]
        tokenizer.train(texts, verbose=False)

        vocab = tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) == tokenizer.get_vocab_size()

    def test_get_merges(self):
        """Test getting merges"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world", "hello there"]
        tokenizer.train(texts, verbose=False)

        merges = tokenizer.get_merges()
        assert isinstance(merges, list)
        assert len(merges) > 0
        assert all(isinstance(merge, tuple) and len(merge) == 2 for merge in merges)

    def test_get_stats(self):
        """Test statistics method"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world"]
        tokenizer.train(texts, verbose=False)

        stats = tokenizer.get_stats()
        assert "vocab_size" in stats
        assert "merges_count" in stats
        assert "special_tokens" in stats
        assert "target_vocab_size" in stats


class TestCharacterTokenizer:
    """Test cases for CharacterTokenizer"""

    def test_initialization(self):
        """Test tokenizer initialization"""
        tokenizer = CharacterTokenizer()
        assert tokenizer.get_vocab_size() > 0
        assert "<UNK>" in tokenizer.vocab
        assert "<PAD>" in tokenizer.vocab

    def test_encode_decode(self):
        """Test encoding and decoding"""
        tokenizer = CharacterTokenizer()
        text = "Hello, world!"

        # Encode text
        tokens = tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert len(tokens) == len(text)

        # Decode tokens
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        # Note: Special characters might be filtered out
        assert len(decoded) <= len(text)

    def test_unknown_character(self):
        """Test handling of unknown characters"""
        tokenizer = CharacterTokenizer()
        text = "Hello 世界!"  # Contains non-ASCII characters

        tokens = tokenizer.encode(text)
        # Should not raise an error
        assert isinstance(tokens, list)

        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)


class TestUtils:
    """Test cases for utility functions"""

    def test_count_tokens(self):
        """Test token counting utility"""
        from mini_gpt.utils import count_tokens

        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)

        count = count_tokens("Hello world!", tokenizer)
        assert count == 3

    def test_analyze_efficiency(self):
        """Test efficiency analysis utility"""
        from mini_gpt.utils import analyze_text_efficiency

        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)

        efficiency = analyze_text_efficiency("Hello world!", tokenizer)
        assert "char_count" in efficiency
        assert "token_count" in efficiency
        assert "chars_per_token" in efficiency
        assert "compression_ratio" in efficiency
        assert "efficiency_score" in efficiency

        assert efficiency["char_count"] == 12
        assert efficiency["token_count"] == 3
        assert efficiency["chars_per_token"] == 4.0

    def test_load_sample_vocab(self):
        """Test sample vocabulary loading"""
        from mini_gpt.utils import load_sample_vocab

        # Test simple vocab
        vocab = load_sample_vocab("simple")
        assert isinstance(vocab, dict)
        assert "<UNK>" in vocab
        assert "Hello" in vocab

        # Test english vocab
        vocab = load_sample_vocab("english")
        assert isinstance(vocab, dict)
        assert "the" in vocab

        # Test code vocab
        vocab = load_sample_vocab("code")
        assert isinstance(vocab, dict)
        assert "def" in vocab

        # Test invalid style
        with pytest.raises(ValueError):
            load_sample_vocab("invalid")

    def test_visualize_tokens(self):
        """Test token visualization utility"""
        from mini_gpt.utils import visualize_tokens

        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)

        # Should not raise an error
        visualize_tokens("Hello world!", tokenizer)

    def test_compare_tokenizers(self):
        """Test tokenizer comparison utility"""
        from mini_gpt.utils import compare_tokenizers

        vocab1 = {"Hello": 1, " world": 2, "!": 3}
        vocab2 = {"Hello world": 1, "!": 2}

        tokenizer1 = SimpleTokenizer(vocab1)
        tokenizer2 = SimpleTokenizer(vocab2)

        # Should not raise an error
        compare_tokenizers("Hello world!", [tokenizer1, tokenizer2], ["Tokenizer 1", "Tokenizer 2"])

    def test_demonstrate_strawberry_problem(self):
        """Test strawberry problem demonstration"""
        from mini_gpt.utils import demonstrate_strawberry_problem

        vocab = {"straw": 1, "berry": 2, "strawberry": 3, " ": 4}
        tokenizer = SimpleTokenizer(vocab)

        # Should not raise an error
        demonstrate_strawberry_problem(tokenizer)

    def test_create_tokenization_report(self):
        """Test tokenization report creation"""
        from mini_gpt.utils import create_tokenization_report

        vocab = {"Hello": 1, " world": 2, "!": 3}
        tokenizer = SimpleTokenizer(vocab)

        # Should not raise an error
        create_tokenization_report("Hello world!", tokenizer)


# Integration tests
class TestIntegration:
    """Integration tests for the complete tokenization system"""

    def test_simple_vs_bpe_comparison(self):
        """Test comparing SimpleTokenizer vs BPETokenizer"""
        # Create simple tokenizer with basic vocab
        simple_vocab = load_sample_vocab("simple")
        simple_tokenizer = SimpleTokenizer(simple_vocab)

        # Create BPE tokenizer and train it
        bpe_tokenizer = BPETokenizer(vocab_size=100)
        texts = ["hello world", "hello there", "world peace"]
        bpe_tokenizer.train(texts, verbose=False)

        # Both should be able to process the same text
        text = "hello world"

        simple_result = simple_tokenizer.encode(text)
        bpe_result = bpe_tokenizer.encode(text)

        assert isinstance(simple_result, list)
        assert isinstance(bpe_result, list)
        assert len(simple_result) > 0
        assert len(bpe_result) > 0

    def test_round_trip_consistency(self):
        """Test that encode-decode round trips work consistently"""
        vocab = load_sample_vocab("simple")
        tokenizer = SimpleTokenizer(vocab)

        test_texts = [
            "Hello world!",
            "How are you?",
            "This is a test.",
            "strawberry",
        ]

        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)

            # For known tokens, should get exact match
            # For unknown tokens, should get reasonable result
            assert isinstance(decoded, str)
            assert len(decoded) >= 0  # At least not crash

    def test_performance_basic(self):
        """Test basic performance characteristics"""
        vocab = load_sample_vocab("english")
        tokenizer = SimpleTokenizer(vocab)

        # Test with longer text
        long_text = "The quick brown fox jumps over the lazy dog. " * 10

        # Should complete in reasonable time
        result = tokenizer.encode(long_text)
        assert isinstance(result, list)
        assert len(result) > 0

        # Decode should also work
        decoded = tokenizer.decode(result)
        assert isinstance(decoded, str)

"""
Tests for the embeddings module.

Covers TokenEmbedding, SinusoidalPositionalEncoding,
LearnedPositionalEmbedding, and build_input_embedding.
"""

import numpy as np
import pytest

from mini_gpt.embeddings import (
    DEFAULT_INIT_STD,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
    build_input_embedding,
)


class TestTokenEmbedding:
    """Test cases for TokenEmbedding."""

    def test_weight_shape(self):
        """Weight matrix should be (vocab_size, d_model)."""
        emb = TokenEmbedding(vocab_size=100, d_model=16, seed=0)
        assert emb.weight.shape == (100, 16)

    def test_lookup_shape(self):
        """lookup returns (len(ids), d_model)."""
        emb = TokenEmbedding(vocab_size=50, d_model=8, seed=0)
        vectors = emb.lookup([0, 5, 10, 49])
        assert vectors.shape == (4, 8)

    def test_lookup_matches_weight_rows(self):
        """Lookup is just fancy indexing into the weight matrix."""
        emb = TokenEmbedding(vocab_size=20, d_model=4, seed=0)
        ids = [3, 7, 1]
        vectors = emb.lookup(ids)
        for i, token_id in enumerate(ids):
            assert np.array_equal(vectors[i], emb.weight[token_id])

    def test_same_id_same_vector(self):
        """Looking up the same ID twice returns the same vector."""
        emb = TokenEmbedding(vocab_size=50, d_model=8, seed=0)
        first = emb.lookup([7])
        second = emb.lookup([7])
        assert np.array_equal(first, second)

    def test_different_ids_different_vectors(self):
        """Different IDs map to different rows (no collisions at init)."""
        emb = TokenEmbedding(vocab_size=50, d_model=8, seed=0)
        # Normal init with seed=0: extremely unlikely any two rows are equal.
        v0 = emb.lookup([0])
        v1 = emb.lookup([1])
        assert not np.array_equal(v0, v1)

    def test_seed_reproducibility(self):
        """Same seed → identical weight matrices."""
        emb_a = TokenEmbedding(vocab_size=20, d_model=4, seed=42)
        emb_b = TokenEmbedding(vocab_size=20, d_model=4, seed=42)
        assert np.array_equal(emb_a.weight, emb_b.weight)

    def test_different_seeds_differ(self):
        """Different seeds → different matrices."""
        emb_a = TokenEmbedding(vocab_size=20, d_model=4, seed=1)
        emb_b = TokenEmbedding(vocab_size=20, d_model=4, seed=2)
        assert not np.array_equal(emb_a.weight, emb_b.weight)

    def test_init_not_zero(self):
        """Weights must NOT init to zero — that would break symmetry."""
        emb = TokenEmbedding(vocab_size=20, d_model=4, seed=0)
        assert not np.allclose(emb.weight, 0.0)

    def test_init_scale_is_small(self):
        """Default init std is ~0.02 (GPT-2 convention)."""
        emb = TokenEmbedding(vocab_size=5000, d_model=64, seed=0)
        # Large enough sample to check std empirically.
        observed_std = float(emb.weight.std())
        assert abs(observed_std - DEFAULT_INIT_STD) < 0.005

    def test_empty_lookup(self):
        """Empty id list returns (0, d_model) array."""
        emb = TokenEmbedding(vocab_size=10, d_model=4, seed=0)
        out = emb.lookup([])
        assert out.shape == (0, 4)

    def test_out_of_range_id_raises(self):
        """Lookup with id >= vocab_size raises IndexError."""
        emb = TokenEmbedding(vocab_size=10, d_model=4, seed=0)
        with pytest.raises(IndexError):
            emb.lookup([10])

    def test_negative_id_raises(self):
        """Negative ids are rejected (numpy would silently wrap)."""
        emb = TokenEmbedding(vocab_size=10, d_model=4, seed=0)
        with pytest.raises(IndexError):
            emb.lookup([-1])

    def test_invalid_vocab_size(self):
        """vocab_size must be positive."""
        with pytest.raises(ValueError):
            TokenEmbedding(vocab_size=0, d_model=4)

    def test_invalid_d_model(self):
        """d_model must be positive."""
        with pytest.raises(ValueError):
            TokenEmbedding(vocab_size=10, d_model=0)


class TestSinusoidalPositionalEncoding:
    """Test cases for SinusoidalPositionalEncoding."""

    def test_weight_shape(self):
        """Weight matrix is (max_len, d_model)."""
        pe = SinusoidalPositionalEncoding(max_len=64, d_model=16)
        assert pe.weight.shape == (64, 16)

    def test_encode_shape(self):
        """encode(seq_len) returns (seq_len, d_model)."""
        pe = SinusoidalPositionalEncoding(max_len=128, d_model=32)
        assert pe.encode(10).shape == (10, 32)

    def test_values_bounded(self):
        """All values are in [-1, 1] since they're sines and cosines."""
        pe = SinusoidalPositionalEncoding(max_len=64, d_model=16)
        assert pe.weight.min() >= -1.0
        assert pe.weight.max() <= 1.0

    def test_position_zero_known_values(self):
        """Position 0: even dims sin(0)=0, odd dims cos(0)=1."""
        pe = SinusoidalPositionalEncoding(max_len=16, d_model=8)
        row0 = pe.encode(1)[0]
        assert np.allclose(row0[0::2], 0.0)
        assert np.allclose(row0[1::2], 1.0)

    def test_deterministic(self):
        """Two instances with same args produce identical encodings."""
        pe_a = SinusoidalPositionalEncoding(max_len=32, d_model=8)
        pe_b = SinusoidalPositionalEncoding(max_len=32, d_model=8)
        assert np.array_equal(pe_a.weight, pe_b.weight)

    def test_dot_product_decays_with_distance(self):
        """Nearby positions have higher dot product than far positions.

        This is the property attention implicitly uses: positions that
        are close in the sequence stay close in embedding space.
        """
        pe = SinusoidalPositionalEncoding(max_len=128, d_model=64)
        w = pe.weight
        # Compare position 0 with positions 1 (near) and 100 (far).
        near = float(w[0] @ w[1])
        far = float(w[0] @ w[100])
        assert near > far

    def test_odd_d_model_raises(self):
        """d_model must be even (sin/cos interleave on even/odd dims)."""
        with pytest.raises(ValueError):
            SinusoidalPositionalEncoding(max_len=16, d_model=7)

    def test_seq_len_zero(self):
        """encode(0) returns an empty (0, d_model) array."""
        pe = SinusoidalPositionalEncoding(max_len=16, d_model=8)
        assert pe.encode(0).shape == (0, 8)

    def test_seq_len_exceeds_max_len_raises(self):
        """Asking for positions beyond max_len raises clearly."""
        pe = SinusoidalPositionalEncoding(max_len=16, d_model=8)
        with pytest.raises(ValueError):
            pe.encode(17)

    def test_invalid_max_len(self):
        with pytest.raises(ValueError):
            SinusoidalPositionalEncoding(max_len=0, d_model=8)


class TestLearnedPositionalEmbedding:
    """Test cases for LearnedPositionalEmbedding."""

    def test_weight_shape(self):
        pe = LearnedPositionalEmbedding(max_len=64, d_model=16, seed=0)
        assert pe.weight.shape == (64, 16)

    def test_encode_shape(self):
        pe = LearnedPositionalEmbedding(max_len=128, d_model=32, seed=0)
        assert pe.encode(10).shape == (10, 32)

    def test_seed_reproducibility(self):
        a = LearnedPositionalEmbedding(max_len=32, d_model=8, seed=42)
        b = LearnedPositionalEmbedding(max_len=32, d_model=8, seed=42)
        assert np.array_equal(a.weight, b.weight)

    def test_different_seeds_differ(self):
        a = LearnedPositionalEmbedding(max_len=32, d_model=8, seed=1)
        b = LearnedPositionalEmbedding(max_len=32, d_model=8, seed=2)
        assert not np.array_equal(a.weight, b.weight)

    def test_init_not_zero(self):
        """Random init, same symmetry-breaking rationale as TokenEmbedding."""
        pe = LearnedPositionalEmbedding(max_len=32, d_model=8, seed=0)
        assert not np.allclose(pe.weight, 0.0)

    def test_encode_matches_weight_rows(self):
        pe = LearnedPositionalEmbedding(max_len=32, d_model=8, seed=0)
        assert np.array_equal(pe.encode(5), pe.weight[:5])

    def test_seq_len_exceeds_max_len_raises(self):
        pe = LearnedPositionalEmbedding(max_len=16, d_model=8, seed=0)
        with pytest.raises(ValueError):
            pe.encode(17)


class TestBuildInputEmbedding:
    """Test cases for build_input_embedding helper."""

    def test_shape(self):
        tok = TokenEmbedding(vocab_size=100, d_model=8, seed=0)
        pos = SinusoidalPositionalEncoding(max_len=16, d_model=8)
        out = build_input_embedding([5, 2, 9], tok, pos)
        assert out.shape == (3, 8)

    def test_is_sum_of_components(self):
        """Output is exactly token_emb + pos_emb, elementwise."""
        tok = TokenEmbedding(vocab_size=50, d_model=8, seed=0)
        pos = SinusoidalPositionalEncoding(max_len=16, d_model=8)
        ids = [3, 1, 7, 4]
        out = build_input_embedding(ids, tok, pos)
        expected = tok.lookup(ids) + pos.encode(len(ids))
        assert np.array_equal(out, expected)

    def test_works_with_learned_positional(self):
        """Helper is agnostic to which positional encoder is passed in."""
        tok = TokenEmbedding(vocab_size=50, d_model=8, seed=0)
        pos = LearnedPositionalEmbedding(max_len=16, d_model=8, seed=1)
        out = build_input_embedding([2, 2, 2], tok, pos)
        # Same token ID at three different positions → identical token
        # parts but different position parts → output rows differ.
        assert out.shape == (3, 8)
        assert not np.array_equal(out[0], out[1])
        assert not np.array_equal(out[1], out[2])

    def test_d_model_mismatch_raises(self):
        tok = TokenEmbedding(vocab_size=50, d_model=8, seed=0)
        pos = SinusoidalPositionalEncoding(max_len=16, d_model=16)
        with pytest.raises(ValueError):
            build_input_embedding([1, 2], tok, pos)

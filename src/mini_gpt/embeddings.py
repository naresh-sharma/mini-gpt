"""
Embedding implementations for MiniGPT
Educational implementations of token and positional embeddings

This module provides three classes that together produce the input vectors
a transformer sees:

1. TokenEmbedding: A (vocab_size, d_model) lookup table. Each token ID
   picks a row — a dense vector — instead of a one-hot.
2. SinusoidalPositionalEncoding: Deterministic sin/cos position signals
   from the original Transformer paper ("Attention Is All You Need").
3. LearnedPositionalEmbedding: The GPT-2 style alternative — positions
   get their own learnable lookup table, same mechanics as TokenEmbedding.

A token's final input vector is token_embedding + positional_encoding.
Both contribute information attention needs: identity (which word) and
order (which position).

All three are NumPy implementations written for clarity, not speed. The
notebook for Part 2 shows that torch.nn.Embedding is doing exactly the
same lookup in production code.
"""

from typing import Optional, Union

import numpy as np

# Standard init scale used by GPT-2 and most GPT-style implementations.
# Small values keep activations well-scaled through a deep stack; learned
# weights drift from this starting point during training. Why NOT zero
# init: every row would have identical gradients and the matrix could
# never learn to distinguish tokens (symmetry breaking).
DEFAULT_INIT_STD = 0.02

# The 10000 base for sinusoidal wavelengths comes from the original
# Transformer paper. It spreads position frequencies from short (high dim)
# to long (low dim), which gives the encoder a multi-scale view of order.
SINUSOIDAL_BASE = 10000.0


class TokenEmbedding:
    """
    A dense lookup table that maps token IDs to vectors.

    Mechanically: a (vocab_size, d_model) matrix where row N is the vector
    for token ID N. `lookup([5, 2, 9])` returns rows 5, 2, 9 stacked. That
    is all `torch.nn.Embedding` does in production code — the rest is
    autograd plumbing.

    The matrix starts as small random numbers. During training, rows
    drift so that semantically similar tokens end up nearby in vector
    space (this is the magic the Part 2 blog post demonstrates with
    pretrained GloVe embeddings).

    Accepts either a single sequence of token IDs (1D) or a batch of
    sequences (2D). The output shape is always the input shape with an
    extra trailing d_model dimension.

    Example:
        >>> emb = TokenEmbedding(vocab_size=1000, d_model=64, seed=42)
        >>> emb.lookup([5, 2, 9]).shape          # single sequence
        (3, 64)
        >>> emb.lookup([[5, 2, 9], [1, 8, 3]]).shape   # batch of 2 sequences
        (2, 3, 64)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        init_std: float = DEFAULT_INIT_STD,
        seed: Optional[int] = None,
    ):
        """
        Initialize a random embedding matrix.

        Args:
            vocab_size: Number of tokens in the vocabulary. Matrix has
                this many rows.
            d_model: Embedding dimension. Each row is a vector of this length.
                Typical values: 64 (tiny), 768 (GPT-2), 12288 (GPT-3).
            init_std: Standard deviation of the normal distribution used
                for initialization. Default matches GPT-2 (0.02).
            seed: Optional seed for reproducible initialization. Useful
                in tests and demos; real training seeds the whole system.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.vocab_size = vocab_size
        self.d_model = d_model

        rng = np.random.default_rng(seed)
        # Normal init (not zeros!) to break symmetry between rows. If every
        # row started equal, every token would receive identical gradients
        # and the matrix could never learn to differentiate them.
        self._weight = rng.normal(loc=0.0, scale=init_std, size=(vocab_size, d_model))

    @property
    def weight(self) -> np.ndarray:
        """The underlying (vocab_size, d_model) matrix. Exposed for
        inspection, visualization, and tests."""
        return self._weight

    def lookup(self, token_ids: Union[list, np.ndarray]) -> np.ndarray:
        """
        Return the embedding vectors for the given token IDs.

        Args:
            token_ids: Integer IDs in range [0, vocab_size). Accepts any
                shape: a flat list/array of length L, or a 2D batch of
                shape (B, L), etc.

        Returns:
            Array of shape ``token_ids.shape + (d_model,)``. A 1D input
            of length L returns (L, d_model); a 2D batch (B, L) returns
            (B, L, d_model).

        Example:
            >>> emb = TokenEmbedding(vocab_size=100, d_model=8, seed=0)
            >>> emb.lookup([0, 5, 99]).shape
            (3, 8)
            >>> emb.lookup([[0, 5], [99, 1]]).shape
            (2, 2, 8)
        """
        ids = np.asarray(token_ids, dtype=np.int64)
        # Range-check only when non-empty: ids.min/max raises on empty arrays.
        if ids.size > 0 and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise IndexError(
                f"token_ids must be in [0, {self.vocab_size}), got range [{ids.min()}, {ids.max()}]"
            )
        # Fancy indexing — this is the entire lookup. NumPy returns an
        # array of shape ids.shape + (d_model,), so batched inputs work
        # out of the box. torch.nn.Embedding does the same operation
        # wrapped in autograd.
        return self._weight[ids]


class SinusoidalPositionalEncoding:
    """
    Deterministic position signals using interleaved sines and cosines.

    From "Attention Is All You Need" (Vaswani et al., 2017). The position
    encoding at position `pos` and dimension `i` is:

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Three useful properties:

    - No parameters to learn — the encoding is a pure function of position.
    - Extrapolates to positions never seen during training (useful for
      longer sequences at inference).
    - The dot product between positions decays with distance, which gives
      attention a built-in notion of "nearness".

    Example:
        >>> pe = SinusoidalPositionalEncoding(max_len=128, d_model=64)
        >>> pe.encode(seq_len=10).shape
        (10, 64)
    """

    def __init__(self, max_len: int, d_model: int):
        """
        Precompute the (max_len, d_model) position matrix.

        Args:
            max_len: Longest sequence length to support.
            d_model: Embedding dimension. Must be even (sin/cos are
                interleaved on even/odd dims).
        """
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError(f"d_model must be a positive even integer, got {d_model}")

        self.max_len = max_len
        self.d_model = d_model

        # position: column vector of position indices (max_len, 1)
        position = np.arange(max_len).reshape(-1, 1)
        # div_term: wavelengths for each even dimension (d_model // 2,)
        # Computed in log-space for numerical stability.
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(SINUSOIDAL_BASE) / d_model))

        pe = np.zeros((max_len, d_model))
        # Even columns get sines, odd columns get cosines. The interleave
        # is what gives each pair of dims a rotating 2D position "clock".
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self._weight = pe

    @property
    def weight(self) -> np.ndarray:
        """The full (max_len, d_model) position matrix."""
        return self._weight

    def encode(self, seq_len: int) -> np.ndarray:
        """
        Return the position encodings for the first `seq_len` positions.

        Args:
            seq_len: Number of positions to return. Must be <= max_len.

        Returns:
            Array of shape (seq_len, d_model).
        """
        if seq_len < 0:
            raise ValueError(f"seq_len must be non-negative, got {seq_len}")
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_len ({self.max_len}). "
                f"Construct SinusoidalPositionalEncoding with a larger max_len."
            )
        return self._weight[:seq_len]


class LearnedPositionalEmbedding:
    """
    Learned positional embeddings (GPT-2 style).

    Mechanically identical to TokenEmbedding: a (max_len, d_model) matrix
    initialized randomly, with position N looking up row N. The model
    learns position representations during training rather than being
    handed them as a fixed function.

    Tradeoff vs SinusoidalPositionalEncoding:

    - Sinusoidal: zero learned parameters, extrapolates past max_len,
      fixed wavelengths whether or not they're optimal for your task.
    - Learned (this): more flexible, can shape itself to the task, but
      cannot extrapolate past max_len and adds max_len * d_model params.

    The original Transformer paper used sinusoidal. GPT-2 and most
    modern LLMs use learned. For tiny educational models either works,
    so this module exposes both and the Part 2 notebook compares them.

    Example:
        >>> pe = LearnedPositionalEmbedding(max_len=128, d_model=64, seed=0)
        >>> pe.encode(seq_len=10).shape
        (10, 64)
    """

    def __init__(
        self,
        max_len: int,
        d_model: int,
        init_std: float = DEFAULT_INIT_STD,
        seed: Optional[int] = None,
    ):
        """
        Initialize a random position matrix.

        Args:
            max_len: Longest sequence length supported.
            d_model: Embedding dimension.
            init_std: Normal init std. Default matches GPT-2 (0.02).
            seed: Optional seed for reproducibility.
        """
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.max_len = max_len
        self.d_model = d_model

        rng = np.random.default_rng(seed)
        self._weight = rng.normal(loc=0.0, scale=init_std, size=(max_len, d_model))

    @property
    def weight(self) -> np.ndarray:
        """The (max_len, d_model) learned position matrix."""
        return self._weight

    def encode(self, seq_len: int) -> np.ndarray:
        """
        Return the position vectors for the first `seq_len` positions.

        Mirrors SinusoidalPositionalEncoding.encode so the two are
        drop-in interchangeable in build_input_embedding.

        Args:
            seq_len: Number of positions to return. Must be <= max_len.

        Returns:
            Array of shape (seq_len, d_model).
        """
        if seq_len < 0:
            raise ValueError(f"seq_len must be non-negative, got {seq_len}")
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_len ({self.max_len}). "
                f"Learned positions beyond max_len were never trained; "
                f"increase max_len at construction."
            )
        return self._weight[:seq_len]


# Types that act as positional encoders (both have the same .encode API).
PositionalEncoder = Union[SinusoidalPositionalEncoding, LearnedPositionalEmbedding]


def build_input_embedding(
    token_ids: Union[list, np.ndarray],
    token_embedding: TokenEmbedding,
    positional_encoder: PositionalEncoder,
) -> np.ndarray:
    """
    Combine token embeddings with positional encodings.

    This is what attention actually receives: each row carries both
    *what* the token is and *where* it sits in the sequence.

    Works on both single sequences and batches. For input shape
    ``(..., seq_len)`` the output is ``(..., seq_len, d_model)``. The
    position vectors broadcast across any leading batch dimensions —
    every item in the batch gets the same positional encoding at
    position i, which is what transformers want.

    Args:
        token_ids: Integer token IDs, 1D (single sequence) or 2D batch.
        token_embedding: A TokenEmbedding matching the vocabulary.
        positional_encoder: Either SinusoidalPositionalEncoding or
            LearnedPositionalEmbedding. Both expose .encode(seq_len).

    Returns:
        Array of shape ``token_ids.shape + (d_model,)``, equal to the
        elementwise sum of the token vectors and the position vectors
        (broadcast across batch).

    Example:
        >>> tok = TokenEmbedding(vocab_size=100, d_model=8, seed=0)
        >>> pos = SinusoidalPositionalEncoding(max_len=16, d_model=8)
        >>> build_input_embedding([5, 2, 9], tok, pos).shape
        (3, 8)
        >>> build_input_embedding([[5, 2, 9], [1, 8, 3]], tok, pos).shape
        (2, 3, 8)
    """
    if token_embedding.d_model != positional_encoder.d_model:
        raise ValueError(
            f"d_model mismatch: token_embedding has {token_embedding.d_model}, "
            f"positional_encoder has {positional_encoder.d_model}"
        )

    token_vectors = token_embedding.lookup(token_ids)
    # Use shape[-2] so this works for both 1D (seq, d_model) and batched
    # (..., seq, d_model) inputs. Scalar inputs (0D) — with shape ()
    # — aren't supported; you'd have a single ID with no sequence dim.
    if token_vectors.ndim < 2:
        raise ValueError(
            "token_ids must have at least one sequence dimension "
            f"(got shape {np.asarray(token_ids).shape})"
        )
    seq_len = token_vectors.shape[-2]
    position_vectors = positional_encoder.encode(seq_len)

    # Elementwise sum with broadcasting: position_vectors is (seq, d_model)
    # and token_vectors is (..., seq, d_model), so positions are added
    # to every item in the batch.
    return token_vectors + position_vectors

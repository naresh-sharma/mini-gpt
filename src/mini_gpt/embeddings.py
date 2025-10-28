"""
Embedding layers for MiniGPT
Converts tokens to meaningful vector representations

This module implements embedding layers that convert discrete tokens
into continuous vector representations that can be processed by neural networks.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that converts token IDs to dense vectors.

    This is the first layer in the transformer that converts discrete
    token IDs into continuous vector representations.

    TODO: Implement in Part 2 - Embeddings notebook
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize the token embedding layer.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the embedding vectors
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # TODO: Initialize embedding layer
        # self.embedding = nn.Embedding(vocab_size, d_model)
        raise NotImplementedError("See Part 2 notebook for implementation")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing token IDs

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) containing embeddings
        """
        raise NotImplementedError("See Part 2 notebook for implementation")


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer that adds position information to embeddings.

    Since transformers don't have inherent notion of position, we need to
    explicitly add positional information to help the model understand
    the order of tokens.

    TODO: Implement in Part 2 - Embeddings notebook
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding layer.

        Args:
            d_model: Dimension of the embedding vectors
            max_len: Maximum sequence length to precompute encodings for
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # TODO: Initialize positional encoding
        # self.register_buffer('pe', self._create_positional_encoding())
        raise NotImplementedError("See Part 2 notebook for implementation")

    def _create_positional_encoding(self) -> torch.Tensor:
        """
        Create the positional encoding matrix.

        Returns:
            Tensor of shape (max_len, d_model) containing positional encodings
        """
        raise NotImplementedError("See Part 2 notebook for implementation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        raise NotImplementedError("See Part 2 notebook for implementation")


class EmbeddingLayer(nn.Module):
    """
    Complete embedding layer combining token and positional embeddings.

    This module combines token embeddings with positional encodings
    to create the final input representation for the transformer.

    TODO: Implement in Part 2 - Embeddings notebook
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000):
        """
        Initialize the embedding layer.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the embedding vectors
            max_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # TODO: Initialize components
        # self.token_embedding = TokenEmbedding(vocab_size, d_model)
        # self.positional_encoding = PositionalEncoding(d_model, max_len)
        # self.dropout = nn.Dropout(0.1)
        raise NotImplementedError("See Part 2 notebook for implementation")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to final embeddings with positional encoding.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing token IDs

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) containing final embeddings
        """
        raise NotImplementedError("See Part 2 notebook for implementation")

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            Dimension of the embedding vectors
        """
        return self.d_model

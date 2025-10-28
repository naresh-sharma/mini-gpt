"""
Attention mechanisms for MiniGPT
The core of transformer architecture

This module implements the attention mechanisms that allow the model
to focus on different parts of the input sequence when processing each token.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    This is the core attention mechanism that computes attention weights
    by taking the dot product of queries and keys, scaled by the square root
    of the dimension.

    TODO: Implement in Part 3 - Attention notebook
    """

    def __init__(self, dropout: float = 0.1):
        """
        Initialize the attention mechanism.

        Args:
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            key: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            value: Value tensor of shape (batch_size, num_heads, seq_len, d_v)
            mask: Optional mask tensor

        Returns:
            Tuple of (output, attention_weights)
        """
        raise NotImplementedError("See Part 3 notebook for implementation")


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    This module implements multi-head attention by running multiple
    attention heads in parallel and concatenating their outputs.

    TODO: Implement in Part 3 - Attention notebook
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: Initialize linear layers for Q, K, V projections
        # self.w_q = nn.Linear(d_model, d_model)
        # self.w_k = nn.Linear(d_model, d_model)
        # self.w_v = nn.Linear(d_model, d_model)
        # self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

        raise NotImplementedError("See Part 3 notebook for implementation")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        raise NotImplementedError("See Part 3 notebook for implementation")


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention for autoregressive language modeling.

    This attention mechanism ensures that each token can only attend
    to previous tokens, making it suitable for language generation.

    TODO: Implement in Part 3 - Attention notebook
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize causal self-attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: Initialize multi-head attention
        # self.multihead_attn = MultiHeadAttention(d_model, num_heads, dropout)
        raise NotImplementedError("See Part 3 notebook for implementation")

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create causal mask to prevent attending to future tokens.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask tensor
        """
        raise NotImplementedError("See Part 3 notebook for implementation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        raise NotImplementedError("See Part 3 notebook for implementation")

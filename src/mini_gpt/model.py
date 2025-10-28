"""
MiniGPT model architecture
Complete transformer implementation

This module implements the full MiniGPT model architecture,
combining all the components into a working language model.
"""

from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.

    This is a simple two-layer MLP with ReLU activation
    that processes each position independently.

    TODO: Implement in Part 4 - Model Architecture notebook
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (usually 4 * d_model)
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # TODO: Initialize linear layers
        # self.linear1 = nn.Linear(d_model, d_ff)
        # self.linear2 = nn.Linear(d_ff, d_model)
        # self.dropout = nn.Dropout(dropout)
        raise NotImplementedError("See Part 4 notebook for implementation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        raise NotImplementedError("See Part 4 notebook for implementation")


class TransformerBlock(nn.Module):
    """
    A single transformer block with self-attention and feed-forward layers.

    This is the basic building block of the transformer architecture,
    containing self-attention, layer normalization, and feed-forward layers.

    TODO: Implement in Part 4 - Model Architecture notebook
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the transformer block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # TODO: Initialize components
        # self.self_attn = CausalSelfAttention(d_model, num_heads, dropout)
        # self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)
        raise NotImplementedError("See Part 4 notebook for implementation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block transformation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        raise NotImplementedError("See Part 4 notebook for implementation")


class MiniGPT(nn.Module):
    """
    MiniGPT model implementation.

    This is the complete MiniGPT model that combines embeddings,
    multiple transformer blocks, and output projection.

    TODO: Implement in Part 4 - Model Architecture notebook
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Initialize the MiniGPT model.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_len = max_len

        # TODO: Initialize components
        # self.embedding = EmbeddingLayer(vocab_size, d_model, max_len)
        # self.transformer_blocks = nn.ModuleList([
        #     TransformerBlock(d_model, num_heads, d_ff, dropout)
        #     for _ in range(num_layers)
        # ])
        # self.layer_norm = nn.LayerNorm(d_model)
        # self.output_projection = nn.Linear(d_model, vocab_size)
        # self.dropout = nn.Dropout(dropout)
        raise NotImplementedError("See Part 4 notebook for implementation")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        raise NotImplementedError("See Part 4 notebook for implementation")

    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            prompt: Initial prompt tokens of shape (batch_size, prompt_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated token IDs of shape (batch_size, max_length)
        """
        raise NotImplementedError("See Part 5 notebook for implementation")

    def get_num_parameters(self) -> int:
        """
        Get the total number of parameters in the model.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """
        Get the model size in megabytes.

        Returns:
            Model size in MB
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / (1024 * 1024)

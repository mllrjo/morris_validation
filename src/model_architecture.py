# src/model_architecture.py
# GPT-style transformer models for Morris validation experiment

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional
from pathlib import Path

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Combined QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using GPT-2 style initialization."""
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        
        return self.out_proj(attn_output)

class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Standard GPT-2 ratio
        
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = nn.GELU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using GPT-2 style initialization."""
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))

class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-normalization (GPT-2 style)
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    """GPT-style transformer model for memorization experiments."""
    
    def __init__(self, vocab_size: int, seq_length: int, d_model: int, 
                 n_layers: int, n_heads: int, d_ff: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_length, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) 
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # Create causal mask
        self.register_buffer("causal_mask", self._create_causal_mask(seq_length))
    
    def _init_weights(self):
        """Initialize all weights using GPT-2 style initialization."""
        # Token and position embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        
        # Layer norm parameters (default initialization is fine)
        nn.init.ones_(self.ln_f.weight)
        nn.init.zeros_(self.ln_f.bias)
        
        # Language model head
        nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def _create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_length, seq_length))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
        
        # Get appropriate mask size
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'total_params': self.get_num_params()
        }

def create_gpt_model(n_layers: int, d_model: int, n_heads: int, 
                    vocab_size: int, seq_length: int, 
                    device: Optional[str] = None) -> GPTModel:
    """Create GPT-style transformer model with specified architecture.
    
    Args:
        n_layers: Number of transformer layers
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        vocab_size: Vocabulary size
        seq_length: Maximum sequence length
        device: Device to place model on (auto-detect if None)
        
    Returns:
        Initialized GPT model
    """
    if device is None:
        device = detect_device()
    
    # Validate parameters
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")
    
    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        seq_length=seq_length,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads
    )
    
    # Move to device
    model = model.to(device)
    
    return model

def count_model_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_config(n_layers: int, d_model: int, n_heads: int, 
                    vocab_size: int, seq_length: int) -> Dict[str, Any]:
    """Get model configuration dictionary.
    
    Args:
        n_layers: Number of transformer layers
        d_model: Model dimension
        n_heads: Number of attention heads  
        vocab_size: Vocabulary size
        seq_length: Maximum sequence length
        
    Returns:
        Configuration dictionary
    """
    # Create temporary model to get parameter count
    temp_model = GPTModel(
        vocab_size=vocab_size,
        seq_length=seq_length,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads
    )
    
    config = {
        'architecture': {
            'n_layers': n_layers,
            'd_model': d_model,
            'n_heads': n_heads,
            'd_head': d_model // n_heads,
            'd_ff': 4 * d_model,  # Standard ratio
            'vocab_size': vocab_size,
            'seq_length': seq_length
        },
        'parameters': {
            'total_params': temp_model.get_num_params(),
            'embedding_params': vocab_size * d_model + seq_length * d_model,
            'transformer_params': temp_model.get_num_params() - (vocab_size * d_model + seq_length * d_model + d_model * vocab_size),
            'output_params': d_model * vocab_size
        },
        'memory_estimate': {
            'params_mb': temp_model.get_num_params() * 4 / (1024 * 1024),  # 4 bytes per float32
            'activations_mb_per_sample': estimate_activation_memory(d_model, n_layers, seq_length)
        }
    }
    
    return config

def detect_device() -> str:
    """Detect best available device (cuda/mps/cpu).
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test MPS device allocation to ensure it actually works
        try:
            test_tensor = torch.zeros(1, device='mps')
            # Test a simple operation
            result = test_tensor + 1
            del test_tensor, result  # Clean up
            return 'mps'
        except (RuntimeError, OSError, Exception):
            # MPS detection succeeded but allocation/operation failed, fall back to CPU
            return 'cpu'
    else:
        return 'cpu'

def estimate_activation_memory(d_model: int, n_layers: int, seq_length: int) -> float:
    """Estimate activation memory usage in MB per sample.
    
    Args:
        d_model: Model dimension
        n_layers: Number of layers
        seq_length: Sequence length
        
    Returns:
        Estimated memory in MB per sample
    """
    # Rough estimate of activation memory
    # Each layer has attention and MLP activations
    attention_memory = seq_length * seq_length * d_model  # Attention matrices
    mlp_memory = seq_length * d_model * 4  # MLP intermediate
    layer_memory = attention_memory + mlp_memory
    
    total_memory = layer_memory * n_layers
    memory_mb = total_memory * 4 / (1024 * 1024)  # 4 bytes per float32
    
    return memory_mb

def create_model_family(base_config: Dict[str, Any], 
                       scale_factors: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Create a family of models with different scales.
    
    Args:
        base_config: Base model configuration
        scale_factors: Dictionary of scale factors for different model sizes
        
    Returns:
        Dictionary mapping model names to their configurations
    """
    model_family = {}
    
    for model_name, scale_factor in scale_factors.items():
        scaled_config = base_config.copy()
        
        # Scale d_model and adjust n_heads to maintain divisibility
        scaled_d_model = int(base_config['d_model'] * scale_factor)
        
        # Ensure d_model is divisible by n_heads
        n_heads = base_config['n_heads']
        while scaled_d_model % n_heads != 0:
            scaled_d_model += 1
        
        scaled_config.update({
            'd_model': scaled_d_model,
            'n_layers': max(1, int(base_config['n_layers'] * scale_factor)),
        })
        
        model_family[model_name] = get_model_config(
            n_layers=scaled_config['n_layers'],
            d_model=scaled_config['d_model'],
            n_heads=scaled_config['n_heads'],
            vocab_size=scaled_config['vocab_size'],
            seq_length=scaled_config['seq_length']
        )
    
    return model_family

def get_morris_model_configs() -> Dict[str, Dict[str, Any]]:
    """Get the specific model configurations for Morris validation experiment.
    
    Returns:
        Dictionary of model configurations matching the experimental design
    """
    # Base configuration for Morris experiment
    base_vocab_size = 2  # Binary sequences
    base_seq_length = 64  # Standard sequence length
    
    # Model architecture configurations
    configs = {
        'nano': {
            'n_layers': 2,
            'd_model': 32,
            'n_heads': 2,
            'vocab_size': base_vocab_size,
            'seq_length': base_seq_length
        },
        'micro': {
            'n_layers': 4,
            'd_model': 64,
            'n_heads': 4,
            'vocab_size': base_vocab_size,
            'seq_length': base_seq_length
        },
        'mini': {
            'n_layers': 6,
            'd_model': 128,
            'n_heads': 8,
            'vocab_size': base_vocab_size,
            'seq_length': base_seq_length
        },
        'small': {
            'n_layers': 8,
            'd_model': 256,
            'n_heads': 16,
            'vocab_size': base_vocab_size,
            'seq_length': base_seq_length
        }
    }
    
    # Generate full configurations with parameter counts
    full_configs = {}
    for name, config in configs.items():
        full_configs[name] = get_model_config(**config)
        full_configs[name]['model_name'] = name
    
    return full_configs

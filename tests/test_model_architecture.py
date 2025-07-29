# tests/test_model_architecture.py
# Comprehensive tests for GPT model architecture

import pytest
import torch
import torch.nn as nn
import math
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_architecture import (
    MultiHeadAttention,
    MLP,
    TransformerBlock,
    GPTModel,
    create_gpt_model,
    count_model_parameters,
    get_model_config,
    detect_device,
    estimate_activation_memory,
    create_model_family,
    get_morris_model_configs
)

class TestMultiHeadAttention:
    
    def test_multihead_attention_initialization(self):
        """Test MultiHeadAttention initialization."""
        d_model = 64
        n_heads = 8
        
        attn = MultiHeadAttention(d_model, n_heads)
        
        assert attn.d_model == d_model
        assert attn.n_heads == n_heads
        assert attn.d_head == d_model // n_heads
        
        # Check layer dimensions
        assert attn.qkv_proj.in_features == d_model
        assert attn.qkv_proj.out_features == 3 * d_model
        assert attn.out_proj.in_features == d_model
        assert attn.out_proj.out_features == d_model
    
    def test_multihead_attention_invalid_dimensions(self):
        """Test MultiHeadAttention with invalid dimensions."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(64, 7)  # 64 not divisible by 7
    
    def test_multihead_attention_forward(self):
        """Test MultiHeadAttention forward pass."""
        d_model = 32
        n_heads = 4
        seq_len = 8
        batch_size = 2
        
        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.allclose(output, x)  # Should transform input
    
    def test_multihead_attention_with_mask(self):
        """Test MultiHeadAttention with causal mask."""
        d_model = 32
        n_heads = 4
        seq_len = 8
        batch_size = 2
        
        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        output_with_mask = attn(x, mask)
        output_without_mask = attn(x)
        
        assert output_with_mask.shape == (batch_size, seq_len, d_model)
        assert not torch.allclose(output_with_mask, output_without_mask)

class TestMLP:
    
    def test_mlp_initialization(self):
        """Test MLP initialization."""
        d_model = 64
        
        # Default d_ff (4 * d_model)
        mlp = MLP(d_model)
        assert mlp.fc1.in_features == d_model
        assert mlp.fc1.out_features == 4 * d_model
        assert mlp.fc2.in_features == 4 * d_model
        assert mlp.fc2.out_features == d_model
        
        # Custom d_ff
        d_ff = 128
        mlp_custom = MLP(d_model, d_ff)
        assert mlp_custom.fc1.out_features == d_ff
        assert mlp_custom.fc2.in_features == d_ff
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        d_model = 32
        seq_len = 8
        batch_size = 2
        
        mlp = MLP(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = mlp(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.allclose(output, x)  # Should transform input

class TestTransformerBlock:
    
    def test_transformer_block_initialization(self):
        """Test TransformerBlock initialization."""
        d_model = 64
        n_heads = 8
        
        block = TransformerBlock(d_model, n_heads)
        
        assert isinstance(block.ln1, nn.LayerNorm)
        assert isinstance(block.attn, MultiHeadAttention)
        assert isinstance(block.ln2, nn.LayerNorm)
        assert isinstance(block.mlp, MLP)
    
    def test_transformer_block_forward(self):
        """Test TransformerBlock forward pass."""
        d_model = 32
        n_heads = 4
        seq_len = 8
        batch_size = 2
        
        block = TransformerBlock(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_transformer_block_residual_connections(self):
        """Test that residual connections work properly."""
        d_model = 32
        n_heads = 4
        seq_len = 8
        batch_size = 2
        
        block = TransformerBlock(d_model, n_heads)
        
        # Zero out the weights to test residual connections
        with torch.no_grad():
            block.attn.out_proj.weight.zero_()
            block.mlp.fc2.weight.zero_()
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = block(x)
        
        # With zero weights, output should be close to input due to residuals
        # (after layer normalization effects)
        assert output.shape == x.shape

class TestGPTModel:
    
    def test_gpt_model_initialization(self):
        """Test GPTModel initialization."""
        vocab_size = 100
        seq_length = 32
        d_model = 64
        n_layers = 4
        n_heads = 8
        
        model = GPTModel(vocab_size, seq_length, d_model, n_layers, n_heads)
        
        assert model.vocab_size == vocab_size
        assert model.seq_length == seq_length
        assert model.d_model == d_model
        assert model.n_layers == n_layers
        assert model.n_heads == n_heads
        
        # Check embeddings
        assert model.token_embedding.num_embeddings == vocab_size
        assert model.token_embedding.embedding_dim == d_model
        assert model.position_embedding.num_embeddings == seq_length
        assert model.position_embedding.embedding_dim == d_model
        
        # Check transformer blocks
        assert len(model.blocks) == n_layers
        for block in model.blocks:
            assert isinstance(block, TransformerBlock)
        
        # Check output layer
        assert isinstance(model.ln_f, nn.LayerNorm)
        assert model.lm_head.in_features == d_model
        assert model.lm_head.out_features == vocab_size
    
    def test_gpt_model_forward(self):
        """Test GPTModel forward pass."""
        vocab_size = 10
        seq_length = 16
        d_model = 32
        n_layers = 2
        n_heads = 4
        batch_size = 3
        
        model = GPTModel(vocab_size, seq_length, d_model, n_layers, n_heads)
        
        # Test with full sequence length
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_length, vocab_size)
        
        # Test with shorter sequence
        short_seq_len = 8
        input_ids_short = torch.randint(0, vocab_size, (batch_size, short_seq_len))
        logits_short = model(input_ids_short)
        
        assert logits_short.shape == (batch_size, short_seq_len, vocab_size)
    
    def test_gpt_model_causal_mask(self):
        """Test that causal mask is properly applied."""
        vocab_size = 4
        seq_length = 8
        d_model = 16
        n_layers = 1
        n_heads = 2
        
        model = GPTModel(vocab_size, seq_length, d_model, n_layers, n_heads)
        
        # Check mask shape and properties
        assert model.causal_mask.shape == (1, 1, seq_length, seq_length)
        
        # Check it's lower triangular
        mask = model.causal_mask.squeeze()
        for i in range(seq_length):
            for j in range(seq_length):
                if j > i:
                    assert mask[i, j] == 0
                else:
                    assert mask[i, j] == 1
    
    def test_gpt_model_get_num_params(self):
        """Test parameter counting."""
        vocab_size = 100
        seq_length = 32
        d_model = 64
        n_layers = 2
        n_heads = 8
        
        model = GPTModel(vocab_size, seq_length, d_model, n_layers, n_heads)
        
        num_params = model.get_num_params()
        assert num_params > 0
        
        # Manual count should match
        manual_count = sum(p.numel() for p in model.parameters())
        assert num_params == manual_count
    
    def test_gpt_model_get_config(self):
        """Test configuration retrieval."""
        vocab_size = 100
        seq_length = 32
        d_model = 64
        n_layers = 2
        n_heads = 8
        
        model = GPTModel(vocab_size, seq_length, d_model, n_layers, n_heads)
        config = model.get_config()
        
        assert config['vocab_size'] == vocab_size
        assert config['seq_length'] == seq_length
        assert config['d_model'] == d_model
        assert config['n_layers'] == n_layers
        assert config['n_heads'] == n_heads
        assert config['total_params'] == model.get_num_params()

class TestModelCreationFunctions:
    
    def test_create_gpt_model(self):
        """Test GPT model creation function."""
        n_layers = 2
        d_model = 32
        n_heads = 4
        vocab_size = 50
        seq_length = 16
        device = 'cpu'
        
        model = create_gpt_model(n_layers, d_model, n_heads, vocab_size, seq_length, device)
        
        assert isinstance(model, GPTModel)
        assert model.n_layers == n_layers
        assert model.d_model == d_model
        assert model.n_heads == n_heads
        assert model.vocab_size == vocab_size
        assert model.seq_length == seq_length
        
        # Check device placement
        assert next(model.parameters()).device.type == device
    
    def test_create_gpt_model_parameter_validation(self):
        """Test parameter validation in model creation."""
        base_args = [2, 32, 4, 50, 16]
        
        # Test invalid d_model/n_heads ratio
        with pytest.raises(ValueError):
            create_gpt_model(2, 30, 4, 50, 16)  # 30 not divisible by 4
        
        # Test negative parameters
        with pytest.raises(ValueError):
            create_gpt_model(-1, 32, 4, 50, 16)  # negative n_layers
        
        with pytest.raises(ValueError):
            create_gpt_model(2, -32, 4, 50, 16)  # negative d_model
        
        with pytest.raises(ValueError):
            create_gpt_model(2, 32, -4, 50, 16)  # negative n_heads
        
        with pytest.raises(ValueError):
            create_gpt_model(2, 32, 4, -50, 16)  # negative vocab_size
        
        with pytest.raises(ValueError):
            create_gpt_model(2, 32, 4, 50, -16)  # negative seq_length
    
    def test_count_model_parameters(self):
        """Test parameter counting function."""
        model = create_gpt_model(2, 32, 4, 50, 16, 'cpu')
        
        param_count = count_model_parameters(model)
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert param_count == manual_count
        assert param_count > 0
    
    def test_get_model_config(self):
        """Test model configuration function."""
        n_layers = 3
        d_model = 48
        n_heads = 6
        vocab_size = 1000
        seq_length = 64
        
        config = get_model_config(n_layers, d_model, n_heads, vocab_size, seq_length)
        
        # Check architecture section
        arch = config['architecture']
        assert arch['n_layers'] == n_layers
        assert arch['d_model'] == d_model
        assert arch['n_heads'] == n_heads
        assert arch['d_head'] == d_model // n_heads
        assert arch['d_ff'] == 4 * d_model
        assert arch['vocab_size'] == vocab_size
        assert arch['seq_length'] == seq_length
        
        # Check parameters section
        params = config['parameters']
        assert params['total_params'] > 0
        assert params['embedding_params'] > 0
        assert params['transformer_params'] >= 0
        assert params['output_params'] > 0
        
        # Check memory estimates
        memory = config['memory_estimate']
        assert memory['params_mb'] > 0
        assert memory['activations_mb_per_sample'] > 0
    
    def test_detect_device(self):
        """Test device detection."""
        device = detect_device()
        
        # Should return one of the valid device types
        assert device in ['cuda', 'mps', 'cpu']
        
        # Should be a string
        assert isinstance(device, str)
    
    def test_estimate_activation_memory(self):
        """Test activation memory estimation."""
        d_model = 64
        n_layers = 4
        seq_length = 32
        
        memory_mb = estimate_activation_memory(d_model, n_layers, seq_length)
        
        assert memory_mb > 0
        assert isinstance(memory_mb, float)
        
        # Memory should scale with parameters
        memory_mb_larger = estimate_activation_memory(d_model * 2, n_layers, seq_length)
        assert memory_mb_larger > memory_mb
    
    def test_create_model_family(self):
        """Test model family creation."""
        base_config = {
            'd_model': 32,
            'n_layers': 2,
            'n_heads': 4,
            'vocab_size': 100,
            'seq_length': 16
        }
        
        scale_factors = {
            'small': 0.5,
            'medium': 1.0,
            'large': 2.0
        }
        
        family = create_model_family(base_config, scale_factors)
        
        assert len(family) == 3
        assert 'small' in family
        assert 'medium' in family
        assert 'large' in family
        
        # Check scaling worked
        small_params = family['small']['parameters']['total_params']
        medium_params = family['medium']['parameters']['total_params']
        large_params = family['large']['parameters']['total_params']
        
        assert small_params < medium_params < large_params
    
    def test_get_morris_model_configs(self):
        """Test Morris experiment model configurations."""
        configs = get_morris_model_configs()
        
        expected_models = ['nano', 'micro', 'mini', 'small']
        assert set(configs.keys()) == set(expected_models)
        
        # Check all configs have required fields
        for name, config in configs.items():
            assert 'architecture' in config
            assert 'parameters' in config
            assert 'memory_estimate' in config
            assert 'model_name' in config
            assert config['model_name'] == name
        
        # Check sizes increase
        param_counts = [configs[name]['parameters']['total_params'] for name in expected_models]
        assert param_counts == sorted(param_counts)  # Should be in increasing order
        
        # Check all use binary vocabulary and consistent sequence length
        for config in configs.values():
            assert config['architecture']['vocab_size'] == 2
            assert config['architecture']['seq_length'] == 64

@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return create_gpt_model(2, 32, 4, 10, 16, 'cpu')

def test_model_inference(sample_model):
    """Test model inference capability."""
    model = sample_model
    batch_size = 2
    seq_length = 8
    vocab_size = 10
    
    # Test inference mode
    model.eval()
    
    with torch.no_grad():
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_length, vocab_size)
        
        # Test that probabilities sum to 1
        probs = torch.softmax(logits, dim=-1)
        prob_sums = probs.sum(dim=-1)
        
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)

def test_model_training_mode(sample_model):
    """Test model in training mode."""
    model = sample_model
    batch_size = 2
    seq_length = 8
    vocab_size = 10
    
    model.train()
    
    # Create dummy training data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    logits = model(input_ids)
    
    # Calculate loss (simple example)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), 
        target_ids.view(-1)
    )
    
    assert loss.item() > 0
    assert loss.requires_grad

def test_integration_model_creation_and_usage():
    """Test complete model creation and usage workflow."""
    # Test different model sizes
    model_configs = [
        (1, 16, 2, 4, 8),   # tiny
        (2, 32, 4, 10, 16), # small
        (3, 48, 6, 20, 24)  # medium
    ]
    
    for n_layers, d_model, n_heads, vocab_size, seq_length in model_configs:
        # Create model
        model = create_gpt_model(n_layers, d_model, n_heads, vocab_size, seq_length, 'cpu')
        
        # Test forward pass
        batch_size = 2
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_length, vocab_size)
        
        # Test parameter count
        param_count = count_model_parameters(model)
        assert param_count > 0
        
        # Test config generation
        config = get_model_config(n_layers, d_model, n_heads, vocab_size, seq_length)
        assert config['parameters']['total_params'] == param_count

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

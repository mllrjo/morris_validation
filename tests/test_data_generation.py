# tests/test_data_generation.py
# Comprehensive tests for data generation functionality

import pytest
import torch
import tempfile
import shutil
import math
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_generation import (
    generate_random_binary_sequences,
    calculate_theoretical_entropy,
    calculate_empirical_entropy,
    generate_dataset_metadata,
    create_dataset_hash,
    save_dataset_cache,
    load_dataset_cache,
    get_or_generate_dataset,
    validate_dataset_properties,
    create_multiple_datasets
)

class TestDataGeneration:
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_generate_random_binary_sequences_basic(self):
        """Test basic random sequence generation."""
        n_sequences = 10
        seq_length = 16
        vocab_size = 2
        seed = 42
        
        dataset = generate_random_binary_sequences(n_sequences, seq_length, vocab_size, seed)
        
        # Check shape
        assert dataset.shape == (n_sequences, seq_length)
        
        # Check data type
        assert dataset.dtype == torch.long
        
        # Check token range
        assert dataset.min() >= 0
        assert dataset.max() < vocab_size
        
        # Check all tokens are valid
        assert torch.all((dataset >= 0) & (dataset < vocab_size))
    
    def test_generate_random_binary_sequences_reproducibility(self):
        """Test that same seed produces same sequences."""
        n_sequences = 5
        seq_length = 8
        seed = 123
        
        dataset1 = generate_random_binary_sequences(n_sequences, seq_length, seed=seed)
        dataset2 = generate_random_binary_sequences(n_sequences, seq_length, seed=seed)
        
        # Should be identical
        assert torch.equal(dataset1, dataset2)
        
        # Different seed should produce different result
        dataset3 = generate_random_binary_sequences(n_sequences, seq_length, seed=seed+1)
        assert not torch.equal(dataset1, dataset3)
    
    def test_generate_random_binary_sequences_different_vocab_sizes(self):
        """Test generation with different vocabulary sizes."""
        n_sequences = 5
        seq_length = 10
        
        for vocab_size in [2, 4, 8, 16]:
            dataset = generate_random_binary_sequences(n_sequences, seq_length, vocab_size)
            
            assert dataset.shape == (n_sequences, seq_length)
            assert dataset.min() >= 0
            assert dataset.max() < vocab_size
    
    def test_generate_random_binary_sequences_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test minimal valid inputs
        dataset = generate_random_binary_sequences(1, 1, 2)
        assert dataset.shape == (1, 1)
        
        # Test error conditions
        with pytest.raises(ValueError):
            generate_random_binary_sequences(0, 10, 2)  # n_sequences <= 0
        
        with pytest.raises(ValueError):
            generate_random_binary_sequences(10, 0, 2)  # seq_length <= 0
        
        with pytest.raises(ValueError):
            generate_random_binary_sequences(10, 10, 1)  # vocab_size < 2
    
    def test_calculate_theoretical_entropy(self):
        """Test theoretical entropy calculation."""
        # Test binary sequences
        dataset = torch.randint(0, 2, (10, 16))  # 10 sequences of 16 tokens
        vocab_size = 2
        
        entropy = calculate_theoretical_entropy(dataset, vocab_size)
        expected_entropy = math.log2(vocab_size) * dataset.numel()
        
        assert abs(entropy - expected_entropy) < 1e-10
        
        # Test different vocab sizes
        for vocab_size in [2, 4, 8]:
            dataset = torch.randint(0, vocab_size, (5, 10))
            entropy = calculate_theoretical_entropy(dataset, vocab_size)
            expected = math.log2(vocab_size) * dataset.numel()
            assert abs(entropy - expected) < 1e-10
        
        # Test empty dataset
        empty_dataset = torch.empty(0, 0, dtype=torch.long)
        entropy = calculate_theoretical_entropy(empty_dataset, 2)
        assert entropy == 0.0
    
    def test_calculate_empirical_entropy(self):
        """Test empirical entropy calculation."""
        # Test perfectly uniform binary data
        # Create dataset with exactly equal token frequencies
        dataset = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])  # 4 zeros, 4 ones
        vocab_size = 2
        
        entropy = calculate_empirical_entropy(dataset, vocab_size)
        # Should be close to theoretical entropy for uniform data
        theoretical = calculate_theoretical_entropy(dataset, vocab_size)
        assert abs(entropy - theoretical) < 1e-10
        
        # Test skewed data
        skewed_dataset = torch.tensor([[0, 0, 0, 1]])  # 3 zeros, 1 one
        entropy = calculate_empirical_entropy(skewed_dataset, 2)
        
        # Manually calculate expected entropy
        p0, p1 = 3/4, 1/4
        expected_per_token = -(p0 * math.log2(p0) + p1 * math.log2(p1))
        expected_total = expected_per_token * 4
        
        assert abs(entropy - expected_total) < 1e-10
        
        # Test empty dataset
        empty_dataset = torch.empty(0, 0, dtype=torch.long)
        entropy = calculate_empirical_entropy(empty_dataset, 2)
        assert entropy == 0.0
    
    def test_generate_dataset_metadata(self):
        """Test dataset metadata generation."""
        n_sequences = 8
        seq_length = 12
        vocab_size = 2
        seed = 789
        
        dataset = generate_random_binary_sequences(n_sequences, seq_length, vocab_size, seed)
        metadata = generate_dataset_metadata(n_sequences, seq_length, vocab_size, seed, dataset)
        
        # Check generation parameters
        gen_params = metadata['generation_params']
        assert gen_params['n_sequences'] == n_sequences
        assert gen_params['seq_length'] == seq_length
        assert gen_params['vocab_size'] == vocab_size
        assert gen_params['seed'] == seed
        
        # Check dataset properties
        props = metadata['dataset_properties']
        assert props['shape'] == [n_sequences, seq_length]
        assert props['total_tokens'] == n_sequences * seq_length
        assert props['bits_per_token_theoretical'] == math.log2(vocab_size)
        
        # Check token statistics
        stats = metadata['token_statistics']
        assert len(stats['token_counts']) == vocab_size
        assert len(stats['token_frequencies']) == vocab_size
        assert stats['unique_tokens'] <= vocab_size
        assert stats['min_token'] >= 0
        assert stats['max_token'] < vocab_size
    
    def test_create_dataset_hash(self):
        """Test dataset hash creation."""
        # Same parameters should give same hash
        hash1 = create_dataset_hash(10, 16, 2, 42)
        hash2 = create_dataset_hash(10, 16, 2, 42)
        assert hash1 == hash2
        
        # Different parameters should give different hashes
        hash3 = create_dataset_hash(10, 16, 2, 43)  # Different seed
        hash4 = create_dataset_hash(11, 16, 2, 42)  # Different n_sequences
        hash5 = create_dataset_hash(10, 17, 2, 42)  # Different seq_length
        hash6 = create_dataset_hash(10, 16, 4, 42)  # Different vocab_size
        
        hashes = [hash1, hash3, hash4, hash5, hash6]
        assert len(set(hashes)) == len(hashes)  # All unique
        
        # Check hash format
        assert len(hash1) == 16  # Truncated SHA256
        assert all(c in '0123456789abcdef' for c in hash1)  # Hex characters
    
    def test_save_load_dataset_cache(self, temp_dir):
        """Test dataset caching functionality."""
        # Generate test dataset
        dataset = generate_random_binary_sequences(5, 8, 2, 42)
        metadata = generate_dataset_metadata(5, 8, 2, 42, dataset)
        
        cache_path = temp_dir / "test_cache.pt"
        
        # Save to cache
        save_dataset_cache(dataset, cache_path, metadata)
        
        # Check file exists
        assert cache_path.exists()
        
        # Load from cache
        loaded_dataset, loaded_metadata = load_dataset_cache(cache_path)
        
        # Check loaded data
        assert loaded_dataset is not None
        assert loaded_metadata is not None
        assert torch.equal(dataset, loaded_dataset)
        assert loaded_metadata == metadata
        
        # Test loading non-existent cache
        nonexistent_path = temp_dir / "nonexistent.pt"
        dataset_none, metadata_none = load_dataset_cache(nonexistent_path)
        assert dataset_none is None
        assert metadata_none is None
    
    def test_get_or_generate_dataset_without_cache(self):
        """Test dataset generation without caching."""
        n_sequences = 6
        seq_length = 10
        vocab_size = 2
        seed = 555
        
        dataset, metadata = get_or_generate_dataset(
            n_sequences, seq_length, vocab_size, seed, 
            cache_dir=None, use_cache=False
        )
        
        # Check dataset properties
        assert dataset.shape == (n_sequences, seq_length)
        assert dataset.min() >= 0
        assert dataset.max() < vocab_size
        
        # Check metadata
        assert metadata['generation_params']['n_sequences'] == n_sequences
        assert metadata['generation_params']['seq_length'] == seq_length
        assert metadata['generation_params']['vocab_size'] == vocab_size
        assert metadata['generation_params']['seed'] == seed
    
    def test_get_or_generate_dataset_with_cache(self, temp_dir):
        """Test dataset generation with caching."""
        n_sequences = 4
        seq_length = 8
        vocab_size = 2
        seed = 666
        
        # First call should generate and cache
        dataset1, metadata1 = get_or_generate_dataset(
            n_sequences, seq_length, vocab_size, seed,
            cache_dir=temp_dir, use_cache=True
        )
        
        # Check cache file was created
        dataset_hash = create_dataset_hash(n_sequences, seq_length, vocab_size, seed)
        cache_path = temp_dir / f"dataset_{dataset_hash}.pt"
        assert cache_path.exists()
        
        # Second call should load from cache
        dataset2, metadata2 = get_or_generate_dataset(
            n_sequences, seq_length, vocab_size, seed,
            cache_dir=temp_dir, use_cache=True
        )
        
        # Should be identical
        assert torch.equal(dataset1, dataset2)
        assert metadata1 == metadata2
    
    def test_validate_dataset_properties(self):
        """Test dataset validation functionality."""
        # Generate valid dataset
        dataset = generate_random_binary_sequences(10, 16, 2, 42)
        expected_shape = (10, 16)
        vocab_size = 2
        
        results = validate_dataset_properties(dataset, expected_shape, vocab_size)
        
        # Should pass all validations for properly generated data
        assert results['shape_correct']
        assert results['dtype_correct']
        assert results['token_range_correct']
        assert results['distribution_uniform']
        assert results['entropy_correct']
        assert results['all_valid']
        
        # Test with wrong shape
        wrong_shape_results = validate_dataset_properties(dataset, (5, 16), vocab_size)
        assert not wrong_shape_results['shape_correct']
        assert not wrong_shape_results['all_valid']
        
        # Test with out-of-range tokens
        invalid_dataset = torch.randint(-1, 3, (10, 16))  # Contains -1 and 2
        invalid_results = validate_dataset_properties(invalid_dataset, expected_shape, vocab_size)
        assert not invalid_results['token_range_correct']
        assert not invalid_results['all_valid']
    
    def test_create_multiple_datasets(self, temp_dir):
        """Test creation of multiple datasets."""
        dataset_sizes = [2, 4, 8]
        seq_length = 12
        vocab_size = 2
        base_seed = 100
        
        datasets = create_multiple_datasets(
            dataset_sizes, seq_length, vocab_size, base_seed, temp_dir
        )
        
        # Check all datasets created
        assert len(datasets) == len(dataset_sizes)
        assert set(datasets.keys()) == set(dataset_sizes)
        
        # Check each dataset
        for size in dataset_sizes:
            dataset, metadata = datasets[size]
            
            # Check shape
            assert dataset.shape == (size, seq_length)
            
            # Check metadata
            assert metadata['generation_params']['n_sequences'] == size
            assert metadata['generation_params']['seq_length'] == seq_length
            assert metadata['generation_params']['vocab_size'] == vocab_size
            
            # Check seed progression
            expected_seed = base_seed + dataset_sizes.index(size) * 1000
            assert metadata['generation_params']['seed'] == expected_seed
        
        # Check datasets are different (different seeds)
        dataset_2, _ = datasets[2]
        dataset_4, _ = datasets[4]
        
        # Compare first 2 sequences (both datasets should have at least 2)
        assert not torch.equal(dataset_2[:2], dataset_4[:2])
    
    def test_entropy_consistency(self):
        """Test consistency between theoretical and empirical entropy for large datasets."""
        # Large dataset for better statistical properties
        n_sequences = 100
        seq_length = 64
        vocab_size = 2
        
        dataset = generate_random_binary_sequences(n_sequences, seq_length, vocab_size, 42)
        
        theoretical_entropy = calculate_theoretical_entropy(dataset, vocab_size)
        empirical_entropy = calculate_empirical_entropy(dataset, vocab_size)
        
        # For large uniform random data, empirical should be close to theoretical
        relative_error = abs(theoretical_entropy - empirical_entropy) / theoretical_entropy
        assert relative_error < 0.05  # Less than 5% error
    
    def test_different_vocab_sizes_entropy(self):
        """Test entropy calculations for different vocabulary sizes."""
        n_sequences = 20
        seq_length = 32
        
        for vocab_size in [2, 4, 8, 16]:
            dataset = generate_random_binary_sequences(n_sequences, seq_length, vocab_size, 42)
            
            theoretical_entropy = calculate_theoretical_entropy(dataset, vocab_size)
            empirical_entropy = calculate_empirical_entropy(dataset, vocab_size)
            
            # Check theoretical entropy formula
            expected_theoretical = math.log2(vocab_size) * dataset.numel()
            assert abs(theoretical_entropy - expected_theoretical) < 1e-10
            
            # Empirical should be reasonably close for uniform random data
            relative_error = abs(theoretical_entropy - empirical_entropy) / theoretical_entropy
            assert relative_error < 0.1  # Less than 10% error

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_integration_data_generation_workflow(temp_dir):
    """Test complete data generation workflow."""
    # Parameters for Morris validation experiment
    dataset_sizes = [1, 2, 4, 8, 16]
    seq_length = 64
    vocab_size = 2
    base_seed = 42
    
    # Generate multiple datasets
    datasets = create_multiple_datasets(
        dataset_sizes, seq_length, vocab_size, base_seed, temp_dir
    )
    
    # Validate each dataset
    for size in dataset_sizes:
        dataset, metadata = datasets[size]
        
        # Validate properties
        expected_shape = (size, seq_length)
        validation_results = validate_dataset_properties(dataset, expected_shape, vocab_size)
        assert validation_results['all_valid'], f"Dataset size {size} failed validation"
        
        # Check metadata completeness
        assert 'generation_params' in metadata
        assert 'dataset_properties' in metadata
        assert 'token_statistics' in metadata
        
        # Verify entropy calculations
        theoretical = metadata['dataset_properties']['theoretical_entropy']
        empirical = metadata['dataset_properties']['empirical_entropy']
        total_tokens = size * seq_length
        
        assert abs(theoretical - math.log2(vocab_size) * total_tokens) < 1e-10
        assert abs(theoretical - empirical) / theoretical < 0.1  # Within 10%
    
    # Test caching worked
    cache_files = list(temp_dir.glob("dataset_*.pt"))
    assert len(cache_files) == len(dataset_sizes)
    
    # Test reproducibility by regenerating
    datasets_2 = create_multiple_datasets(
        dataset_sizes, seq_length, vocab_size, base_seed, temp_dir
    )
    
    for size in dataset_sizes:
        dataset_1, _ = datasets[size]
        dataset_2, _ = datasets_2[size]
        assert torch.equal(dataset_1, dataset_2), f"Dataset size {size} not reproducible"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

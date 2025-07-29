# src/data_generation.py
# Random binary sequence generation for Morris validation experiment

import torch
import numpy as np
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

def generate_random_binary_sequences(n_sequences: int, seq_length: int, 
                                    vocab_size: int = 2, seed: int = 42) -> torch.Tensor:
    """Generate random binary sequences for memorization testing.
    
    Args:
        n_sequences: Number of sequences to generate
        seq_length: Length of each sequence in tokens
        vocab_size: Vocabulary size (default 2 for binary)
        seed: Random seed for reproducibility
        
    Returns:
        Tensor of shape (n_sequences, seq_length) with random token IDs
    """
    if n_sequences <= 0:
        raise ValueError("n_sequences must be positive")
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate uniform random sequences
    sequences = torch.randint(
        low=0, 
        high=vocab_size, 
        size=(n_sequences, seq_length),
        dtype=torch.long
    )
    
    return sequences

def calculate_theoretical_entropy(dataset: torch.Tensor, vocab_size: int) -> float:
    """Calculate theoretical entropy H(X) for uniform random data.
    
    Args:
        dataset: Tensor containing token sequences
        vocab_size: Size of vocabulary
        
    Returns:
        Theoretical entropy in bits
    """
    if dataset.numel() == 0:
        return 0.0
    
    # For uniform random data, entropy is log2(vocab_size) per token
    bits_per_token = math.log2(vocab_size)
    total_tokens = dataset.numel()
    total_entropy = bits_per_token * total_tokens
    
    return total_entropy

def calculate_empirical_entropy(dataset: torch.Tensor, vocab_size: int) -> float:
    """Calculate empirical entropy from actual token frequencies.
    
    Args:
        dataset: Tensor containing token sequences
        vocab_size: Size of vocabulary
        
    Returns:
        Empirical entropy in bits
    """
    if dataset.numel() == 0:
        return 0.0
    
    # Flatten dataset and count token frequencies
    flat_data = dataset.flatten()
    total_tokens = len(flat_data)
    
    # Check for invalid tokens (negative or >= vocab_size)
    min_token = flat_data.min().item()
    max_token = flat_data.max().item()
    
    if min_token < 0 or max_token >= vocab_size:
        # Return NaN for invalid token ranges
        return float('nan')
    
    # Count frequencies for each token
    token_counts = torch.bincount(flat_data, minlength=vocab_size)
    token_probs = token_counts.float() / total_tokens
    
    # Calculate entropy: H = -sum(p * log2(p))
    entropy = 0.0
    for prob in token_probs:
        if prob > 0:  # Avoid log(0)
            entropy -= prob * math.log2(prob)
    
    # Total entropy for all tokens
    total_entropy = entropy * total_tokens
    
    return total_entropy

def generate_dataset_metadata(n_sequences: int, seq_length: int, vocab_size: int, 
                            seed: int, dataset: torch.Tensor) -> Dict[str, Any]:
    """Generate comprehensive metadata for a dataset.
    
    Args:
        n_sequences: Number of sequences generated
        seq_length: Length of each sequence
        vocab_size: Vocabulary size used
        seed: Random seed used
        dataset: Generated dataset tensor
        
    Returns:
        Dictionary containing dataset metadata
    """
    theoretical_entropy = calculate_theoretical_entropy(dataset, vocab_size)
    empirical_entropy = calculate_empirical_entropy(dataset, vocab_size)
    
    # Calculate basic statistics
    flat_data = dataset.flatten()
    token_counts = torch.bincount(flat_data, minlength=vocab_size)
    
    metadata = {
        'generation_params': {
            'n_sequences': n_sequences,
            'seq_length': seq_length,
            'vocab_size': vocab_size,
            'seed': seed
        },
        'dataset_properties': {
            'shape': list(dataset.shape),
            'total_tokens': dataset.numel(),
            'theoretical_entropy': theoretical_entropy,
            'empirical_entropy': empirical_entropy,
            'entropy_deviation': abs(theoretical_entropy - empirical_entropy),
            'bits_per_token_theoretical': math.log2(vocab_size),
            'bits_per_token_empirical': empirical_entropy / dataset.numel() if dataset.numel() > 0 else 0.0
        },
        'token_statistics': {
            'token_counts': token_counts.tolist(),
            'token_frequencies': (token_counts.float() / dataset.numel()).tolist(),
            'unique_tokens': (token_counts > 0).sum().item(),
            'min_token': flat_data.min().item(),
            'max_token': flat_data.max().item()
        }
    }
    
    return metadata

def create_dataset_hash(n_sequences: int, seq_length: int, vocab_size: int, seed: int) -> str:
    """Create unique hash for dataset parameters.
    
    Args:
        n_sequences: Number of sequences
        seq_length: Length of each sequence
        vocab_size: Vocabulary size
        seed: Random seed
        
    Returns:
        SHA256 hash string for dataset parameters
    """
    param_string = f"{n_sequences}_{seq_length}_{vocab_size}_{seed}"
    return hashlib.sha256(param_string.encode()).hexdigest()[:16]

def save_dataset_cache(dataset: torch.Tensor, cache_path: Path, 
                      metadata: Dict[str, Any]) -> None:
    """Save generated dataset to cache with metadata.
    
    Args:
        dataset: Generated dataset tensor
        cache_path: Path to save cached dataset
        metadata: Dataset generation metadata
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset and metadata
    cache_data = {
        'dataset': dataset,
        'metadata': metadata,
        'cache_version': '1.0'
    }
    
    torch.save(cache_data, cache_path)

def load_dataset_cache(cache_path: Path) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """Load cached dataset and metadata.
    
    Args:
        cache_path: Path to cached dataset
        
    Returns:
        Tuple of (dataset_tensor, metadata_dict) or (None, None) if not found
    """
    cache_path = Path(cache_path)
    
    if not cache_path.exists():
        return None, None
    
    try:
        cache_data = torch.load(cache_path, map_location='cpu')
        
        # Validate cache format
        if not isinstance(cache_data, dict):
            return None, None
        
        if 'dataset' not in cache_data or 'metadata' not in cache_data:
            return None, None
        
        return cache_data['dataset'], cache_data['metadata']
        
    except (RuntimeError, KeyError, TypeError):
        return None, None

def get_or_generate_dataset(n_sequences: int, seq_length: int, vocab_size: int = 2,
                          seed: int = 42, cache_dir: Optional[Path] = None,
                          use_cache: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Get dataset from cache or generate new one.
    
    Args:
        n_sequences: Number of sequences to generate
        seq_length: Length of each sequence
        vocab_size: Vocabulary size (default 2 for binary)
        seed: Random seed for reproducibility
        cache_dir: Directory for caching datasets
        use_cache: Whether to use caching
        
    Returns:
        Tuple of (dataset_tensor, metadata_dict)
    """
    # Try to load from cache first
    if use_cache and cache_dir is not None:
        cache_dir = Path(cache_dir)
        dataset_hash = create_dataset_hash(n_sequences, seq_length, vocab_size, seed)
        cache_path = cache_dir / f"dataset_{dataset_hash}.pt"
        
        cached_dataset, cached_metadata = load_dataset_cache(cache_path)
        if cached_dataset is not None and cached_metadata is not None:
            return cached_dataset, cached_metadata
    
    # Generate new dataset
    dataset = generate_random_binary_sequences(n_sequences, seq_length, vocab_size, seed)
    metadata = generate_dataset_metadata(n_sequences, seq_length, vocab_size, seed, dataset)
    
    # Save to cache
    if use_cache and cache_dir is not None:
        cache_dir = Path(cache_dir)
        dataset_hash = create_dataset_hash(n_sequences, seq_length, vocab_size, seed)
        cache_path = cache_dir / f"dataset_{dataset_hash}.pt"
        save_dataset_cache(dataset, cache_path, metadata)
    
    return dataset, metadata

def validate_dataset_properties(dataset: torch.Tensor, expected_shape: Tuple[int, int],
                               vocab_size: int, tolerance: float = 0.05) -> Dict[str, bool]:
    """Validate that generated dataset has expected properties.
    
    Args:
        dataset: Generated dataset tensor
        expected_shape: Expected (n_sequences, seq_length) shape
        vocab_size: Expected vocabulary size
        tolerance: Tolerance for statistical tests (default 5%)
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    # Check shape
    results['shape_correct'] = tuple(dataset.shape) == expected_shape
    
    # Check data type
    results['dtype_correct'] = dataset.dtype == torch.long
    
    # Check token range
    min_token = dataset.min().item()
    max_token = dataset.max().item()
    results['token_range_correct'] = (min_token >= 0) and (max_token < vocab_size)
    
    # Check if tokens are roughly uniformly distributed
    flat_data = dataset.flatten()
    
    # Only check distribution if tokens are in valid range
    if results['token_range_correct']:
        token_counts = torch.bincount(flat_data, minlength=vocab_size)
        expected_count = len(flat_data) / vocab_size
        
        # Chi-square goodness of fit test (simplified)
        max_deviation = max(abs(count - expected_count) for count in token_counts)
        results['distribution_uniform'] = max_deviation < (expected_count * tolerance * 3)
        
        # Check entropy is close to theoretical
        theoretical_entropy = calculate_theoretical_entropy(dataset, vocab_size)
        empirical_entropy = calculate_empirical_entropy(dataset, vocab_size)
        
        if math.isnan(empirical_entropy) or theoretical_entropy == 0:
            results['entropy_correct'] = False
        else:
            entropy_error = abs(theoretical_entropy - empirical_entropy) / theoretical_entropy
            results['entropy_correct'] = entropy_error < tolerance
    else:
        # Skip distribution and entropy checks for invalid token ranges
        results['distribution_uniform'] = False
        results['entropy_correct'] = False
    
    # Overall validation
    results['all_valid'] = all(results.values())
    
    return results

def create_multiple_datasets(dataset_sizes: List[int], seq_length: int,
                           vocab_size: int = 2, base_seed: int = 42,
                           cache_dir: Optional[Path] = None) -> Dict[int, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Create multiple datasets of different sizes with related seeds.
    
    Args:
        dataset_sizes: List of dataset sizes (number of sequences)
        seq_length: Length of each sequence
        vocab_size: Vocabulary size (default 2 for binary)
        base_seed: Base seed for reproducibility
        cache_dir: Directory for caching datasets
        
    Returns:
        Dictionary mapping dataset_size to (dataset_tensor, metadata_dict)
    """
    datasets = {}
    
    for i, size in enumerate(dataset_sizes):
        # Use different but deterministic seed for each dataset
        seed = base_seed + i * 1000
        
        dataset, metadata = get_or_generate_dataset(
            n_sequences=size,
            seq_length=seq_length,
            vocab_size=vocab_size,
            seed=seed,
            cache_dir=cache_dir,
            use_cache=True
        )
        
        datasets[size] = (dataset, metadata)
    
    return datasets

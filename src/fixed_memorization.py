# src/fixed_memorization.py
# Fixed dataset memorization for meaningful Morris validation

import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from data_generation import (
    generate_random_binary_sequences, 
    generate_dataset_metadata,
    save_dataset_cache,
    load_dataset_cache
)

def generate_fixed_memorization_dataset(
    n_sequences: int,
    seq_length: int,
    vocab_size: int = 2,
    seed: int = 42,
    cache_dir: Optional[Path] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Generate a fixed dataset for memorization experiments.
    
    This creates a specific set of sequences that the model will train on
    repeatedly across many epochs to enable memorization.
    
    Args:
        n_sequences: Number of unique sequences to memorize
        seq_length: Length of each sequence
        vocab_size: Vocabulary size (2 for binary)
        seed: Random seed for reproducible sequence generation
        cache_dir: Directory for caching the fixed dataset
        
    Returns:
        Tuple of (dataset_tensor, metadata_dict)
    """
    cache_id = f"fixed_memo_{n_sequences}_{seq_length}_{vocab_size}_{seed}"
    
    # Try to load from cache first
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / f"{cache_id}.pt"
        
        cached_dataset, cached_metadata = load_dataset_cache(cache_path)
        if cached_dataset is not None:
            print(f"📂 Loaded fixed memorization dataset from cache: {cache_path}")
            return cached_dataset, cached_metadata
    
    print(f"🎲 Generating fixed memorization dataset: {n_sequences} sequences")
    
    # Generate the fixed sequences
    dataset = generate_random_binary_sequences(n_sequences, seq_length, vocab_size, seed)
    
    # Create metadata with memorization-specific information
    metadata = generate_dataset_metadata(n_sequences, seq_length, vocab_size, seed, dataset)
    
    # Add memorization-specific metadata
    metadata['memorization_info'] = {
        'dataset_type': 'fixed_memorization',
        'unique_sequences': n_sequences,
        'memorization_target': f"Model should memorize all {n_sequences} sequences",
        'expected_behavior': "Morris memorization should increase with training epochs",
        'cache_id': cache_id
    }
    
    # Calculate theoretical maximum memorization
    # Each sequence contributes seq_length bits of information
    total_bits_to_memorize = n_sequences * seq_length
    metadata['memorization_info']['theoretical_max_memorization'] = total_bits_to_memorize
    
    # Save to cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_id}.pt"
        save_dataset_cache(dataset, cache_path, metadata)
        print(f"💾 Cached fixed dataset to: {cache_path}")
    
    return dataset, metadata

def create_memorization_training_config(
    model_name: str = 'nano',
    n_sequences: int = 500,
    seq_length: int = 64,
    vocab_size: int = 2,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    eval_interval: int = 200,
    memorization_eval_interval: int = 500,
    device: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Create configuration optimized for fixed dataset memorization.
    
    Args:
        model_name: Model size ('nano', 'micro', 'mini', 'small')
        n_sequences: Number of unique sequences to memorize
        seq_length: Length of each sequence
        vocab_size: Vocabulary size (2 for binary)
        epochs: Number of complete passes through the fixed dataset
        learning_rate: Learning rate for memorization
        batch_size: Training batch size
        eval_interval: Steps between loss evaluations
        memorization_eval_interval: Steps between memorization measurements
        device: Device to use (auto-detect if None)
        seed: Random seed
        
    Returns:
        Configuration dictionary optimized for memorization
    """
    from model_architecture import detect_device, get_morris_model_configs
    
    if device is None:
        device = detect_device()
    
    # Get model configuration
    model_configs = get_morris_model_configs()
    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model_config = model_configs[model_name]
    
    # Calculate training steps
    # Each epoch = one pass through all sequences
    steps_per_epoch = max(1, n_sequences // batch_size)
    max_steps = epochs * steps_per_epoch
    
    # Memorization-optimized hyperparameters
    config = {
        'model': {
            'name': model_name,
            'n_layers': model_config['architecture']['n_layers'],
            'd_model': model_config['architecture']['d_model'],
            'n_heads': model_config['architecture']['n_heads'],
            'vocab_size': vocab_size,
            'seq_length': seq_length,
            'total_params': model_config['parameters']['total_params']
        },
        'data': {
            'dataset_size': n_sequences,
            'eval_dataset_size': min(100, n_sequences // 5),  # 20% for evaluation
            'seq_length': seq_length,
            'vocab_size': vocab_size,
            'seed': seed,
            'memorization_mode': True,
            'epochs': epochs,
            'steps_per_epoch': steps_per_epoch
        },
        'training': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_steps': max_steps,
            'warmup_steps': min(100, max_steps // 10),  # 10% warmup
            'weight_decay': 0.0,  # No regularization for memorization
            'grad_clip_norm': 1.0,
            'device': device,
            'epochs': epochs
        },
        'evaluation': {
            'eval_interval': eval_interval,
            'memorization_eval_interval': memorization_eval_interval,
            'save_checkpoint_interval': max(500, max_steps // 10)
        },
        'experiment': {
            'seed': seed,
            'experiment_type': 'fixed_memorization',
            'created_time': None  # Will be set by caller
        }
    }
    
    return config

def estimate_memorization_capacity_for_model(model_name: str, n_sequences: int, 
                                           seq_length: int) -> Dict[str, Any]:
    """Estimate memorization capacity and requirements for a model.
    
    Args:
        model_name: Model size
        n_sequences: Number of sequences to memorize  
        seq_length: Length of each sequence
        
    Returns:
        Capacity analysis dictionary
    """
    from model_architecture import get_morris_model_configs
    
    model_configs = get_morris_model_configs()
    if model_name not in model_configs:
        return {'error': f'Unknown model: {model_name}'}
    
    model_config = model_configs[model_name]
    model_params = model_config['parameters']['total_params']
    
    # Information to memorize
    total_bits_to_memorize = n_sequences * seq_length
    
    # Morris scaling expectations
    expected_capacity_at_3_6 = model_params * 3.6  # bits
    expected_capacity_at_1_0 = model_params * 1.0  # bits (conservative)
    
    # Analysis
    analysis = {
        'model_name': model_name,
        'model_parameters': model_params,
        'sequences_to_memorize': n_sequences,
        'bits_per_sequence': seq_length,
        'total_bits_to_memorize': total_bits_to_memorize,
        'expected_capacity_3_6': expected_capacity_at_3_6,
        'expected_capacity_1_0': expected_capacity_at_1_0,
        'memorization_ratio_3_6': total_bits_to_memorize / expected_capacity_at_3_6,
        'memorization_ratio_1_0': total_bits_to_memorize / expected_capacity_at_1_0,
        'recommendation': None
    }
    
    # Recommendations based on ratios
    if analysis['memorization_ratio_1_0'] < 0.1:
        analysis['recommendation'] = f"EASY - Model should memorize perfectly ({analysis['memorization_ratio_1_0']:.1%} of capacity)"
    elif analysis['memorization_ratio_1_0'] < 0.5:
        analysis['recommendation'] = f"MODERATE - Good memorization expected ({analysis['memorization_ratio_1_0']:.1%} of capacity)"
    elif analysis['memorization_ratio_1_0'] < 1.0:
        analysis['recommendation'] = f"CHALLENGING - Partial memorization ({analysis['memorization_ratio_1_0']:.1%} of capacity)"
    else:
        analysis['recommendation'] = f"DIFFICULT - May exceed capacity ({analysis['memorization_ratio_1_0']:.1%} of capacity)"
    
    return analysis

def suggest_optimal_memorization_experiments() -> Dict[str, Dict[str, Any]]:
    """Suggest optimal fixed memorization experiments for each model size.
    
    Returns:
        Dictionary mapping model names to suggested experiment parameters
    """
    suggestions = {
        'nano': {
            'n_sequences': [200, 500, 1000],
            'reasoning': "Small model - start with few sequences for clear memorization",
            'expected_bits_per_param': [0.5, 1.2, 2.4],
            'epochs': 200,
            'learning_rate': 0.002
        },
        'micro': {
            'n_sequences': [1000, 2000, 4000], 
            'reasoning': "Medium model - moderate sequence counts",
            'expected_bits_per_param': [0.3, 0.6, 1.3],
            'epochs': 150,
            'learning_rate': 0.001
        },
        'mini': {
            'n_sequences': [3000, 6000, 12000],
            'reasoning': "Large model - needs many sequences to approach capacity",
            'expected_bits_per_param': [0.2, 0.3, 0.6],
            'epochs': 100,
            'learning_rate': 0.0008
        },
        'small': {
            'n_sequences': [8000, 15000, 30000],
            'reasoning': "Very large model - massive datasets needed",
            'expected_bits_per_param': [0.05, 0.1, 0.2],
            'epochs': 80,
            'learning_rate': 0.0005
        }
    }
    
    # Add capacity analysis for each suggestion
    for model_name, suggestion in suggestions.items():
        suggestion['capacity_analysis'] = []
        for n_seq in suggestion['n_sequences']:
            analysis = estimate_memorization_capacity_for_model(model_name, n_seq, 64)
            suggestion['capacity_analysis'].append(analysis)
    
    return suggestions

def print_memorization_experiment_plan():
    """Print a comprehensive plan for fixed memorization experiments."""
    print("🎯 Fixed Dataset Memorization Experiment Plan")
    print("=" * 60)
    
    suggestions = suggest_optimal_memorization_experiments()
    
    for model_name, suggestion in suggestions.items():
        print(f"\n📊 {model_name.upper()} Model Plan:")
        print(f"   Sequences: {suggestion['n_sequences']}")
        print(f"   Epochs: {suggestion['epochs']}")
        print(f"   Learning rate: {suggestion['learning_rate']}")
        print(f"   Reasoning: {suggestion['reasoning']}")
        print(f"   Expected bits/param: {suggestion['expected_bits_per_param']}")
        
        print(f"   Capacity Analysis:")
        for i, analysis in enumerate(suggestion['capacity_analysis']):
            n_seq = suggestion['n_sequences'][i]
            rec = analysis['recommendation']
            print(f"     {n_seq:,} sequences: {rec}")

if __name__ == "__main__":
    print_memorization_experiment_plan()

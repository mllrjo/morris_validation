# function_declarations.py
# Central registry of all functions with type annotations and descriptions

from typing import Dict, List, Any, Optional, Tuple
import torch
from pathlib import Path
import json

# =============================================================================
# LOGGING_UTILS.PY
# =============================================================================

def setup_logging_directories(base_path: Path) -> Dict[str, Path]:
    """Create logging directory structure and return paths.
    
    Args:
        base_path: Root directory for logs
        
    Returns:
        Dictionary mapping log types to their directory paths
    """
    pass

def generate_experiment_id() -> str:
    """Generate unique experiment ID using timestamp and random component.
    
    Returns:
        Unique experiment identifier string
    """
    pass

def log_experiment_metadata(experiment_id: str, metadata: Dict[str, Any], 
                          log_dir: Path) -> None:
    """Log experiment configuration and metadata to JSON file.
    
    Args:
        experiment_id: Unique experiment identifier
        metadata: Dictionary containing experiment configuration
        log_dir: Directory to save metadata file
    """
    pass

def log_metrics_csv(experiment_id: str, metrics: Dict[str, Any], 
                   log_dir: Path, append: bool = True) -> None:
    """Log metrics to CSV file with atomic writes.
    
    Args:
        experiment_id: Unique experiment identifier
        metrics: Dictionary of metric values to log
        log_dir: Directory to save CSV file
        append: Whether to append to existing file or create new
    """
    pass

def log_training_step(experiment_id: str, step: int, loss: float, 
                     morris_bits: float, log_dir: Path) -> None:
    """Log individual training step metrics.
    
    Args:
        experiment_id: Unique experiment identifier
        step: Training step number
        loss: Training loss value
        morris_bits: Morris memorization in bits
        log_dir: Directory to save training logs
    """
    pass

def atomic_write_json(data: Dict[str, Any], file_path: Path) -> None:
    """Write JSON data atomically to prevent corruption.
    
    Args:
        data: Dictionary to write as JSON
        file_path: Target file path
    """
    pass

# =============================================================================
# CHECKPOINT_MANAGER.PY
# =============================================================================

def save_experiment_state(experiment_id: str, state: Dict[str, Any], 
                         checkpoint_dir: Path) -> None:
    """Save complete experiment state for resumability.
    
    Args:
        experiment_id: Unique experiment identifier
        state: Complete experiment state dictionary
        checkpoint_dir: Directory to save checkpoint files
    """
    pass

def load_experiment_state(experiment_id: str, 
                         checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """Load experiment state from checkpoint.
    
    Args:
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Experiment state dictionary if found, None otherwise
    """
    pass

def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         step: int, experiment_id: str, 
                         checkpoint_dir: Path) -> None:
    """Save model and optimizer state during training.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        step: Current training step
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory to save model checkpoints
    """
    pass

def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         experiment_id: str, checkpoint_dir: Path,
                         step: Optional[int] = None) -> Tuple[int, bool]:
    """Load model and optimizer state from checkpoint.
    
    Args:
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory containing model checkpoints
        step: Specific step to load (latest if None)
        
    Returns:
        Tuple of (loaded_step, success_flag)
    """
    pass

def find_incomplete_experiments(checkpoint_dir: Path) -> List[str]:
    """Find experiments that can be resumed.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        List of experiment IDs that can be resumed
    """
    pass

# =============================================================================
# DATA_GENERATION.PY  
# =============================================================================

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
    pass

def calculate_theoretical_entropy(dataset: torch.Tensor, vocab_size: int) -> float:
    """Calculate theoretical entropy H(X) for uniform random data.
    
    Args:
        dataset: Tensor containing token sequences
        vocab_size: Size of vocabulary
        
    Returns:
        Theoretical entropy in bits
    """
    pass

def save_dataset_cache(dataset: torch.Tensor, cache_path: Path, 
                      metadata: Dict[str, Any]) -> None:
    """Save generated dataset to cache with metadata.
    
    Args:
        dataset: Generated dataset tensor
        cache_path: Path to save cached dataset
        metadata: Dataset generation metadata
    """
    pass

def load_dataset_cache(cache_path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load cached dataset and metadata.
    
    Args:
        cache_path: Path to cached dataset
        
    Returns:
        Tuple of (dataset_tensor, metadata_dict)
    """
    pass

# =============================================================================
# MODEL_ARCHITECTURE.PY
# =============================================================================

def create_gpt_model(n_layers: int, d_model: int, n_heads: int, 
                    vocab_size: int, seq_length: int, 
                    device: Optional[str] = None) -> torch.nn.Module:
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
    pass

def count_model_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    pass

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
    pass

def detect_device() -> str:
    """Detect best available device (cuda/mps/cpu).
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    pass

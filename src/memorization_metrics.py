# src/memorization_metrics.py
# Morris memorization measurement H(X) - H(X|θ̂) for validation experiment

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time

from model_architecture import detect_device, count_model_parameters
from data_generation import calculate_theoretical_entropy
from logging_utils import log_metrics_csv, log_training_step

def calculate_model_conditional_entropy(model: torch.nn.Module, dataset: torch.Tensor, 
                                      device: str, batch_size: int = 32) -> float:
    """Calculate H(X|θ̂) using model's cross-entropy loss on dataset.
    
    Args:
        model: Trained PyTorch model
        dataset: Dataset tensor of shape (n_sequences, seq_length)
        device: Device to run computation on
        batch_size: Batch size for memory-efficient computation
        
    Returns:
        Conditional entropy H(X|θ̂) in bits
    """
    if dataset.numel() == 0:
        return 0.0
    
    model.eval()
    model = model.to(device)
    dataset = dataset.to(device)
    
    total_log_likelihood = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_data = dataset[i:batch_end]
            
            # Model forward pass
            logits = model(batch_data)  # (batch_size, seq_length, vocab_size)
            
            # Calculate per-token cross-entropy
            # Shift for autoregressive prediction
            input_ids = batch_data[:, :-1]  # (batch_size, seq_length-1)
            target_ids = batch_data[:, 1:]  # (batch_size, seq_length-1)
            logits = logits[:, :-1, :]      # (batch_size, seq_length-1, vocab_size)
            
            # Reshape for cross-entropy calculation
            flat_logits = logits.reshape(-1, logits.size(-1))  # (batch_size * (seq_length-1), vocab_size)
            flat_targets = target_ids.reshape(-1)              # (batch_size * (seq_length-1))
            
            # Calculate log-probabilities
            log_probs = F.log_softmax(flat_logits, dim=-1)
            token_log_likelihoods = log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
            
            # Accumulate
            total_log_likelihood += token_log_likelihoods.sum().item()
            total_tokens += flat_targets.numel()
    
    if total_tokens == 0:
        return 0.0
    
    # Convert from nats to bits and compute entropy
    # H(X|θ̂) = -E[log P(X|θ̂)] = -average log likelihood
    conditional_entropy_nats = -total_log_likelihood / total_tokens
    conditional_entropy_bits = conditional_entropy_nats / math.log(2)
    
    # Total entropy for all tokens
    total_conditional_entropy = conditional_entropy_bits * total_tokens
    
    return total_conditional_entropy

def calculate_morris_memorization(theoretical_entropy: float, 
                                conditional_entropy: float) -> float:
    """Calculate Morris memorization H(X) - H(X|θ̂).
    
    Args:
        theoretical_entropy: Theoretical entropy H(X) in bits
        conditional_entropy: Model conditional entropy H(X|θ̂) in bits
        
    Returns:
        Morris memorization in bits
    """
    if math.isnan(theoretical_entropy) or math.isnan(conditional_entropy):
        return float('nan')
    
    if theoretical_entropy < 0 or conditional_entropy < 0:
        return float('nan')
    
    # Morris memorization = H(X) - H(X|θ̂)
    memorization = theoretical_entropy - conditional_entropy
    
    # Memorization should be non-negative in theory
    # But numerical precision might cause small negative values
    return max(0.0, memorization)

def evaluate_memorization_on_dataset(model: torch.nn.Module, dataset: torch.Tensor,
                                    metadata: Dict[str, Any], device: Optional[str] = None,
                                    batch_size: int = 32) -> Dict[str, Any]:
    """Evaluate Morris memorization on a complete dataset.
    
    Args:
        model: Trained PyTorch model
        dataset: Dataset tensor
        metadata: Dataset metadata from data_generation
        device: Device to use (auto-detect if None)
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing memorization metrics
    """
    if device is None:
        device = detect_device()
    
    # Get theoretical entropy from metadata or calculate
    if 'dataset_properties' in metadata and 'theoretical_entropy' in metadata['dataset_properties']:
        theoretical_entropy = metadata['dataset_properties']['theoretical_entropy']
    else:
        vocab_size = metadata['generation_params']['vocab_size']
        theoretical_entropy = calculate_theoretical_entropy(dataset, vocab_size)
    
    # Calculate conditional entropy
    conditional_entropy = calculate_model_conditional_entropy(model, dataset, device, batch_size)
    
    # Calculate Morris memorization
    morris_memorization = calculate_morris_memorization(theoretical_entropy, conditional_entropy)
    
    # Additional metrics
    model_params = count_model_parameters(model)
    bits_per_parameter = morris_memorization / model_params if model_params > 0 else 0.0
    
    total_tokens = dataset.numel()
    bits_per_token = morris_memorization / total_tokens if total_tokens > 0 else 0.0
    
    # Memorization efficiency metrics
    memorization_fraction = morris_memorization / theoretical_entropy if theoretical_entropy > 0 else 0.0
    
    results = {
        'theoretical_entropy_bits': theoretical_entropy,
        'conditional_entropy_bits': conditional_entropy,
        'morris_memorization_bits': morris_memorization,
        'model_parameters': model_params,
        'bits_per_parameter': bits_per_parameter,
        'bits_per_token': bits_per_token,
        'memorization_fraction': memorization_fraction,
        'total_tokens': total_tokens,
        'dataset_size': len(dataset),
        'evaluation_device': device
    }
    
    return results

def track_memorization_during_training(model: torch.nn.Module, dataset: torch.Tensor,
                                      metadata: Dict[str, Any], current_step: int,
                                      experiment_id: str, log_dir: Path,
                                      device: Optional[str] = None,
                                      batch_size: int = 32) -> Dict[str, Any]:
    """Track memorization during training and log results.
    
    Args:
        model: Current training model
        dataset: Evaluation dataset
        metadata: Dataset metadata
        current_step: Current training step
        experiment_id: Experiment identifier for logging
        log_dir: Directory for logging
        device: Device to use (auto-detect if None)
        batch_size: Batch size for evaluation
        
    Returns:
        Memorization metrics dictionary
    """
    # Evaluate memorization
    metrics = evaluate_memorization_on_dataset(model, dataset, metadata, device, batch_size)
    
    # Add training context
    metrics['training_step'] = current_step
    metrics['timestamp'] = time.time()
    
    # Log to training step logs (using existing function)
    # Note: This assumes we have a loss value available - will be 0 for memorization-only tracking
    morris_bits = metrics['morris_memorization_bits']
    log_training_step(experiment_id, current_step, 0.0, morris_bits, log_dir)
    
    # Log detailed metrics
    detailed_metrics = {
        f'step_{current_step}_memorization': metrics,
        'experiment_id': experiment_id
    }
    log_metrics_csv(experiment_id, detailed_metrics, log_dir, append=True)
    
    return metrics

def compute_bits_per_parameter(memorization_bits: float, model_parameters: int) -> float:
    """Compute memorization efficiency in bits per parameter.
    
    Args:
        memorization_bits: Total memorization in bits
        model_parameters: Number of model parameters
        
    Returns:
        Bits per parameter ratio
    """
    if model_parameters <= 0:
        return 0.0
    
    if math.isnan(memorization_bits) or memorization_bits < 0:
        return 0.0
    
    return memorization_bits / model_parameters

def validate_memorization_bounds(memorization_bits: float, theoretical_entropy: float,
                                conditional_entropy: float) -> Dict[str, bool]:
    """Validate that memorization metrics satisfy theoretical bounds.
    
    Args:
        memorization_bits: Calculated memorization
        theoretical_entropy: Theoretical entropy H(X)
        conditional_entropy: Conditional entropy H(X|θ̂)
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    # Basic validity checks
    results['memorization_non_negative'] = memorization_bits >= 0
    results['conditional_entropy_non_negative'] = conditional_entropy >= 0
    results['theoretical_entropy_non_negative'] = theoretical_entropy >= 0
    
    # Theoretical bounds
    # H(X|θ̂) ≤ H(X) always (conditioning reduces entropy)
    results['conditional_le_theoretical'] = conditional_entropy <= theoretical_entropy + 1e-6  # Small tolerance
    
    # Morris memorization should not exceed theoretical entropy
    results['memorization_le_theoretical'] = memorization_bits <= theoretical_entropy + 1e-6
    
    # Consistency check: H(X) - H(X|θ̂) = memorization
    expected_memorization = theoretical_entropy - conditional_entropy
    memorization_error = abs(memorization_bits - expected_memorization)
    results['memorization_consistent'] = memorization_error < 1e-6
    
    # Overall validity
    results['all_valid'] = all(results.values())
    
    return results

def compute_memorization_per_layer(model: torch.nn.Module, dataset: torch.Tensor,
                                  device: str, batch_size: int = 32) -> Dict[str, float]:
    """Compute memorization contribution by layer depth (approximation).
    
    Args:
        model: Trained model
        dataset: Evaluation dataset
        device: Computation device
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary mapping layer indices to approximate memorization contributions
    """
    if not hasattr(model, 'blocks'):
        return {'total': calculate_model_conditional_entropy(model, dataset, device, batch_size)}
    
    model.eval()
    model = model.to(device)
    dataset = dataset.to(device)
    
    layer_entropies = {}
    
    # Progressive evaluation through layers
    with torch.no_grad():
        for layer_idx in range(len(model.blocks) + 1):
            # Create a temporary model that stops at layer_idx
            if layer_idx == 0:
                # Just embeddings
                def partial_forward(x):
                    batch_size, seq_len = x.shape
                    position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                    
                    token_embeds = model.token_embedding(x)
                    pos_embeds = model.position_embedding(position_ids)
                    return token_embeds + pos_embeds
            else:
                # Through layer_idx-1 transformer blocks
                def partial_forward(x):
                    batch_size, seq_len = x.shape
                    position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                    
                    token_embeds = model.token_embedding(x)
                    pos_embeds = model.position_embedding(position_ids)
                    x = token_embeds + pos_embeds
                    
                    mask = model.causal_mask[:, :, :seq_len, :seq_len]
                    
                    for i in range(min(layer_idx, len(model.blocks))):
                        x = model.blocks[i](x, mask)
                    
                    return x
            
            # This is a simplified approximation - full implementation would require
            # more sophisticated techniques to measure layer-wise contributions
            layer_entropies[f'layer_{layer_idx}'] = 0.0  # Placeholder
    
    return layer_entropies

def create_memorization_report(evaluation_results: List[Dict[str, Any]],
                              model_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive memorization analysis report.
    
    Args:
        evaluation_results: List of evaluation results for different models/datasets
        model_configs: Dictionary of model configurations
        
    Returns:
        Comprehensive analysis report
    """
    report = {
        'summary': {},
        'model_analysis': {},
        'scaling_analysis': {},
        'validation_results': {}
    }
    
    if not evaluation_results:
        return report
    
    # Summary statistics
    memorizations = [r['morris_memorization_bits'] for r in evaluation_results]
    bits_per_param = [r['bits_per_parameter'] for r in evaluation_results]
    model_sizes = [r['model_parameters'] for r in evaluation_results]
    
    report['summary'] = {
        'total_evaluations': len(evaluation_results),
        'memorization_range': [min(memorizations), max(memorizations)],
        'bits_per_param_range': [min(bits_per_param), max(bits_per_param)],
        'model_size_range': [min(model_sizes), max(model_sizes)],
        'average_memorization': np.mean(memorizations),
        'average_bits_per_param': np.mean(bits_per_param)
    }
    
    # Model-wise analysis
    for i, result in enumerate(evaluation_results):
        model_id = f"model_{i}"
        report['model_analysis'][model_id] = {
            'memorization_bits': result['morris_memorization_bits'],
            'bits_per_parameter': result['bits_per_parameter'],
            'memorization_fraction': result['memorization_fraction'],
            'model_parameters': result['model_parameters']
        }
    
    # Scaling law analysis (if multiple model sizes)
    if len(set(model_sizes)) > 1:
        # Fit power law: memorization ~ parameters^α
        log_params = np.log(model_sizes)
        log_memorization = np.log(memorizations)
        
        # Simple linear regression in log space
        A = np.vstack([log_params, np.ones(len(log_params))]).T
        scaling_coeff, log_intercept = np.linalg.lstsq(A, log_memorization, rcond=None)[0]
        
        report['scaling_analysis'] = {
            'scaling_exponent': scaling_coeff,
            'log_intercept': log_intercept,
            'theoretical_exponent': 1.0,  # Expected linear scaling
            'morris_3_6_bits_validation': np.mean(bits_per_param)  # Should approach 3.6
        }
    
    # Validation against theoretical bounds
    validation_passes = 0
    for result in evaluation_results:
        bounds_check = validate_memorization_bounds(
            result['morris_memorization_bits'],
            result['theoretical_entropy_bits'],
            result['conditional_entropy_bits']
        )
        if bounds_check['all_valid']:
            validation_passes += 1
    
    report['validation_results'] = {
        'bounds_validation_pass_rate': validation_passes / len(evaluation_results),
        'total_evaluations': len(evaluation_results),
        'passed_validations': validation_passes
    }
    
    return report

def estimate_memorization_capacity(model_parameters: int, bits_per_param: float = 3.6) -> float:
    """Estimate theoretical memorization capacity based on Morris scaling law.
    
    Args:
        model_parameters: Number of model parameters
        bits_per_param: Expected bits per parameter (default 3.6 from Morris et al.)
        
    Returns:
        Estimated memorization capacity in bits
    """
    if model_parameters <= 0 or bits_per_param <= 0:
        return 0.0
    
    return model_parameters * bits_per_param

def memorization_efficiency_score(actual_memorization: float, 
                                 model_parameters: int,
                                 expected_bits_per_param: float = 3.6) -> float:
    """Calculate memorization efficiency relative to Morris scaling law.
    
    Args:
        actual_memorization: Measured memorization in bits
        model_parameters: Number of model parameters
        expected_bits_per_param: Expected efficiency (default 3.6)
        
    Returns:
        Efficiency score (1.0 = perfect efficiency, >1.0 = above expected)
    """
    if model_parameters <= 0 or expected_bits_per_param <= 0:
        return 0.0
    
    expected_memorization = estimate_memorization_capacity(model_parameters, expected_bits_per_param)
    
    if expected_memorization <= 0:
        return 0.0
    
    return actual_memorization / expected_memorization

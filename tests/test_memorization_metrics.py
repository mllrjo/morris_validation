# tests/test_memorization_metrics.py
# Comprehensive tests for Morris memorization measurement

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import math
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from memorization_metrics import (
    calculate_model_conditional_entropy,
    calculate_morris_memorization,
    evaluate_memorization_on_dataset,
    track_memorization_during_training,
    compute_bits_per_parameter,
    validate_memorization_bounds,
    compute_memorization_per_layer,
    create_memorization_report,
    estimate_memorization_capacity,
    memorization_efficiency_score
)

from model_architecture import create_gpt_model, detect_device
from data_generation import generate_random_binary_sequences, generate_dataset_metadata

class SimplePerfectMemoryModel(nn.Module):
    """Model that perfectly memorizes its training data for testing."""
    
    def __init__(self, vocab_size=2, seq_length=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        # Return very confident predictions (high logits)
        output = torch.zeros(batch_size, seq_len, self.vocab_size)
        
        # For perfect autoregressive prediction: predict the actual next token
        for i in range(batch_size):
            for j in range(seq_len):
                if j < seq_len - 1:  # For all positions except the last
                    next_token = x[i, j+1].item()  # Get the actual next token
                    # Very high logit for the correct token
                    output[i, j, next_token] = 10.0
                    # Very low logit for other tokens
                    for k in range(self.vocab_size):
                        if k != next_token:
                            output[i, j, k] = -10.0
                else:  # For the last position, predict token 0 (arbitrary)
                    output[i, j, 0] = 10.0
                    for k in range(1, self.vocab_size):
                        output[i, j, k] = -10.0
        
        return output

class SimpleRandomModel(nn.Module):
    """Model that produces random predictions for testing."""
    
    def __init__(self, vocab_size=2, seq_length=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(seq_length, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        # Random predictions (uniform distribution)
        output = torch.zeros(batch_size, seq_len, self.vocab_size)
        return output  # Uniform logits (0, 0) = uniform probabilities after softmax

class TestMemorizationMetrics:
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return 'cpu'  # Force CPU for reproducible tests
    
    @pytest.fixture
    def test_dataset(self):
        """Create test dataset with known properties."""
        dataset = generate_random_binary_sequences(4, 8, 2, 42)
        metadata = generate_dataset_metadata(4, 8, 2, 42, dataset)
        return dataset, metadata
    
    def test_calculate_model_conditional_entropy_perfect_memory(self, device, test_dataset):
        """Test conditional entropy calculation with perfect memory model."""
        dataset, metadata = test_dataset
        
        # Create model that gives perfect predictions
        model = SimplePerfectMemoryModel(vocab_size=2, seq_length=8)
        
        conditional_entropy = calculate_model_conditional_entropy(model, dataset, device, batch_size=2)
        
        # Perfect memory should have very low conditional entropy
        assert conditional_entropy >= 0.0
        # With perfect predictions and high confidence (logit=10), entropy should be very small
        # Each token contributes -log_2(softmax(10)) ≈ -log_2(0.99995) ≈ 0.00007 bits
        # For 4 sequences of 7 tokens each (28 total tokens), total should be < 0.01 bits
        assert conditional_entropy < 0.01, f"Expected very low entropy for perfect model, got {conditional_entropy}"
    
    def test_calculate_model_conditional_entropy_random_predictions(self, device, test_dataset):
        """Test conditional entropy with random prediction model."""
        dataset, metadata = test_dataset
        
        # Model with random predictions
        model = SimpleRandomModel(vocab_size=2, seq_length=8)
        
        conditional_entropy = calculate_model_conditional_entropy(model, dataset, device, batch_size=2)
        
        # Random predictions should have entropy close to theoretical maximum
        theoretical_entropy = metadata['dataset_properties']['theoretical_entropy']
        
        # Should be positive and reasonably high for random model
        assert conditional_entropy >= 0.0
        # For uniform random model, conditional entropy should be close to theoretical
        # (within reasonable bounds due to finite sample size)
        assert conditional_entropy > theoretical_entropy * 0.5
    
    def test_calculate_model_conditional_entropy_edge_cases(self, device):
        """Test conditional entropy calculation edge cases."""
        model = SimpleRandomModel()
        
        # Empty dataset
        empty_dataset = torch.empty(0, 0, dtype=torch.long)
        entropy = calculate_model_conditional_entropy(model, empty_dataset, device)
        assert entropy == 0.0
        
        # Single token dataset
        single_dataset = torch.tensor([[0]], dtype=torch.long)
        entropy = calculate_model_conditional_entropy(model, single_dataset, device)
        assert entropy >= 0.0
    
    def test_calculate_morris_memorization_basic(self):
        """Test basic Morris memorization calculation."""
        theoretical_entropy = 100.0
        conditional_entropy = 60.0
        
        memorization = calculate_morris_memorization(theoretical_entropy, conditional_entropy)
        
        assert memorization == 40.0  # 100 - 60
        assert memorization >= 0.0
    
    def test_calculate_morris_memorization_edge_cases(self):
        """Test Morris memorization edge cases."""
        # Perfect memorization (conditional entropy = 0)
        memorization = calculate_morris_memorization(100.0, 0.0)
        assert memorization == 100.0
        
        # No memorization (conditional = theoretical)
        memorization = calculate_morris_memorization(100.0, 100.0)
        assert memorization == 0.0
        
        # Invalid inputs (NaN)
        memorization = calculate_morris_memorization(float('nan'), 50.0)
        assert math.isnan(memorization)
        
        memorization = calculate_morris_memorization(100.0, float('nan'))
        assert math.isnan(memorization)
        
        # Negative inputs
        memorization = calculate_morris_memorization(-10.0, 50.0)
        assert math.isnan(memorization)
        
        # Numerical precision case (slightly negative memorization)
        memorization = calculate_morris_memorization(50.0, 50.1)  # Small negative
        assert memorization == 0.0  # Should be clipped to 0
    
    def test_evaluate_memorization_on_dataset(self, device, test_dataset):
        """Test complete dataset memorization evaluation."""
        dataset, metadata = test_dataset
        
        # Use a simple model for testing
        model = SimpleRandomModel(vocab_size=2, seq_length=8)
        
        results = evaluate_memorization_on_dataset(model, dataset, metadata, device, batch_size=2)
        
        # Check all required fields present
        required_fields = [
            'theoretical_entropy_bits', 'conditional_entropy_bits', 'morris_memorization_bits',
            'model_parameters', 'bits_per_parameter', 'bits_per_token',
            'memorization_fraction', 'total_tokens', 'dataset_size', 'evaluation_device'
        ]
        
        for field in required_fields:
            assert field in results
        
        # Check value validity
        assert results['theoretical_entropy_bits'] >= 0
        assert results['conditional_entropy_bits'] >= 0
        assert results['morris_memorization_bits'] >= 0
        assert results['model_parameters'] > 0
        assert results['bits_per_parameter'] >= 0
        assert results['total_tokens'] == dataset.numel()
        assert results['dataset_size'] == len(dataset)
        assert results['evaluation_device'] == device
    
    def test_track_memorization_during_training(self, temp_dir, device, test_dataset):
        """Test memorization tracking during training."""
        dataset, metadata = test_dataset
        model = SimpleRandomModel()
        experiment_id = "test_tracking"
        current_step = 100
        
        metrics = track_memorization_during_training(
            model, dataset, metadata, current_step, experiment_id, temp_dir, device
        )
        
        # Check metrics include training context
        assert 'training_step' in metrics
        assert 'timestamp' in metrics
        assert metrics['training_step'] == current_step
        
        # Check logging files created
        metrics_file = temp_dir / f"{experiment_id}_metrics.csv"
        assert metrics_file.exists()
    
    def test_compute_bits_per_parameter(self):
        """Test bits per parameter computation."""
        # Normal case
        bits_per_param = compute_bits_per_parameter(1000.0, 100)
        assert bits_per_param == 10.0
        
        # Edge cases
        assert compute_bits_per_parameter(100.0, 0) == 0.0  # Zero parameters
        assert compute_bits_per_parameter(0.0, 100) == 0.0  # Zero memorization
        assert compute_bits_per_parameter(float('nan'), 100) == 0.0  # NaN input
        assert compute_bits_per_parameter(-100.0, 100) == 0.0  # Negative memorization
    
    def test_validate_memorization_bounds(self):
        """Test memorization bounds validation."""
        # Valid case
        results = validate_memorization_bounds(40.0, 100.0, 60.0)
        assert results['all_valid']
        assert results['memorization_non_negative']
        assert results['conditional_le_theoretical']
        assert results['memorization_le_theoretical']
        assert results['memorization_consistent']
        
        # Invalid case: conditional > theoretical
        results = validate_memorization_bounds(0.0, 50.0, 100.0)
        assert not results['conditional_le_theoretical']
        assert not results['all_valid']
        
        # Invalid case: memorization > theoretical
        results = validate_memorization_bounds(150.0, 100.0, 50.0)
        assert not results['memorization_le_theoretical']
        assert not results['all_valid']
        
        # Invalid case: negative values
        results = validate_memorization_bounds(-10.0, 100.0, 60.0)
        assert not results['memorization_non_negative']
        assert not results['all_valid']
    
    def test_compute_memorization_per_layer_simple_model(self, device, test_dataset):
        """Test per-layer memorization with simple model."""
        dataset, _ = test_dataset
        model = SimpleRandomModel()
        
        layer_results = compute_memorization_per_layer(model, dataset, device)
        
        # Simple model should return total only
        assert 'total' in layer_results
        assert layer_results['total'] >= 0.0
    
    def test_compute_memorization_per_layer_transformer(self, device):
        """Test per-layer memorization with transformer model."""
        # Create small transformer
        model = create_gpt_model(n_layers=2, d_model=32, n_heads=2, vocab_size=2, seq_length=16, device=device)
        dataset = generate_random_binary_sequences(2, 16, 2, 42)
        
        layer_results = compute_memorization_per_layer(model, dataset, device, batch_size=1)
        
        # Should have layer-wise breakdown
        assert 'layer_0' in layer_results
        assert 'layer_1' in layer_results
        assert 'layer_2' in layer_results
    
    def test_create_memorization_report(self):
        """Test memorization report generation."""
        # Mock evaluation results
        evaluation_results = [
            {
                'morris_memorization_bits': 100.0,
                'bits_per_parameter': 2.0,
                'memorization_fraction': 0.5,
                'model_parameters': 50,
                'theoretical_entropy_bits': 200.0,
                'conditional_entropy_bits': 100.0
            },
            {
                'morris_memorization_bits': 400.0,
                'bits_per_parameter': 4.0,
                'memorization_fraction': 0.8,
                'model_parameters': 100,
                'theoretical_entropy_bits': 500.0,
                'conditional_entropy_bits': 100.0
            }
        ]
        
        model_configs = {}  # Not used in current implementation
        
        report = create_memorization_report(evaluation_results, model_configs)
        
        # Check report structure
        assert 'summary' in report
        assert 'model_analysis' in report
        assert 'scaling_analysis' in report
        assert 'validation_results' in report
        
        # Check summary content
        summary = report['summary']
        assert summary['total_evaluations'] == 2
        assert summary['memorization_range'] == [100.0, 400.0]
        assert summary['bits_per_param_range'] == [2.0, 4.0]
        assert summary['average_memorization'] == 250.0
        assert summary['average_bits_per_param'] == 3.0
        
        # Check scaling analysis
        scaling = report['scaling_analysis']
        assert 'scaling_exponent' in scaling
        assert 'morris_3_6_bits_validation' in scaling
    
    def test_create_memorization_report_empty(self):
        """Test memorization report with empty results."""
        report = create_memorization_report([], {})
        
        assert 'summary' in report
        assert 'model_analysis' in report
        assert 'scaling_analysis' in report
        assert 'validation_results' in report
    
    def test_estimate_memorization_capacity(self):
        """Test memorization capacity estimation."""
        # Standard case
        capacity = estimate_memorization_capacity(1000, 3.6)
        assert capacity == 3600.0
        
        # Edge cases
        assert estimate_memorization_capacity(0, 3.6) == 0.0
        assert estimate_memorization_capacity(1000, 0) == 0.0
        assert estimate_memorization_capacity(-100, 3.6) == 0.0
    
    def test_memorization_efficiency_score(self):
        """Test memorization efficiency scoring."""
        # Perfect efficiency (actual = expected)
        score = memorization_efficiency_score(3600.0, 1000, 3.6)
        assert abs(score - 1.0) < 1e-10
        
        # Above expected efficiency
        score = memorization_efficiency_score(7200.0, 1000, 3.6)
        assert abs(score - 2.0) < 1e-10
        
        # Below expected efficiency
        score = memorization_efficiency_score(1800.0, 1000, 3.6)
        assert abs(score - 0.5) < 1e-10
        
        # Edge cases
        assert memorization_efficiency_score(1000.0, 0, 3.6) == 0.0
        assert memorization_efficiency_score(1000.0, 100, 0) == 0.0
    
    def test_integration_with_real_transformer(self, device):
        """Test memorization measurement with real transformer model."""
        # Create small transformer model
        model = create_gpt_model(
            n_layers=2, d_model=32, n_heads=2, 
            vocab_size=2, seq_length=16, device=device
        )
        
        # Generate test dataset
        dataset = generate_random_binary_sequences(8, 16, 2, 42)
        metadata = generate_dataset_metadata(8, 16, 2, 42, dataset)
        
        # Evaluate memorization
        results = evaluate_memorization_on_dataset(model, dataset, metadata, device, batch_size=4)
        
        # Validate results make sense
        assert results['morris_memorization_bits'] >= 0
        assert results['bits_per_parameter'] >= 0
        assert results['memorization_fraction'] >= 0
        assert results['memorization_fraction'] <= 1.0
        
        # Validate bounds
        bounds_check = validate_memorization_bounds(
            results['morris_memorization_bits'],
            results['theoretical_entropy_bits'],
            results['conditional_entropy_bits']
        )
        assert bounds_check['all_valid'], f"Bounds validation failed: {bounds_check}"
    
    def test_memorization_consistency_across_batch_sizes(self, device, test_dataset):
        """Test that memorization is consistent across different batch sizes."""
        dataset, metadata = test_dataset
        model = SimpleRandomModel()
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4]
        results = []
        
        for batch_size in batch_sizes:
            result = evaluate_memorization_on_dataset(model, dataset, metadata, device, batch_size)
            results.append(result['morris_memorization_bits'])
        
        # Results should be very similar (within small numerical tolerance)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-6, f"Batch size {batch_sizes[i]} gave different result"
    
    def test_memorization_scaling_properties(self):
        """Test memorization scaling properties."""
        # Test with different model sizes
        model_sizes = [100, 500, 1000]
        memorizations = [360, 1800, 3600]  # 3.6 bits per parameter
        
        # Calculate efficiency scores
        scores = [memorization_efficiency_score(mem, size, 3.6) for mem, size in zip(memorizations, model_sizes)]
        
        # All should be close to 1.0 (perfect efficiency)
        for score in scores:
            assert abs(score - 1.0) < 1e-10
        
        # Test capacity estimation
        for size in model_sizes:
            capacity = estimate_memorization_capacity(size, 3.6)
            assert capacity == size * 3.6

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_integration_memorization_workflow(temp_dir):
    """Test complete memorization measurement workflow."""
    device = 'cpu'
    
    # Create model and dataset
    model = create_gpt_model(n_layers=2, d_model=32, n_heads=2, vocab_size=2, seq_length=16, device=device)
    dataset = generate_random_binary_sequences(16, 16, 2, 42)
    metadata = generate_dataset_metadata(16, 16, 2, 42, dataset)
    
    # Test memorization evaluation
    results = evaluate_memorization_on_dataset(model, dataset, metadata, device)
    
    # Test training tracking
    experiment_id = "integration_test"
    tracking_results = track_memorization_during_training(
        model, dataset, metadata, 50, experiment_id, temp_dir, device
    )
    
    # Test bounds validation
    bounds_check = validate_memorization_bounds(
        results['morris_memorization_bits'],
        results['theoretical_entropy_bits'], 
        results['conditional_entropy_bits']
    )
    
    # All validations should pass
    assert bounds_check['all_valid']
    assert results['morris_memorization_bits'] >= 0
    assert tracking_results['training_step'] == 50
    
    # Test report generation
    evaluation_list = [results]
    report = create_memorization_report(evaluation_list, {})
    assert report['summary']['total_evaluations'] == 1
    assert report['validation_results']['bounds_validation_pass_rate'] == 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

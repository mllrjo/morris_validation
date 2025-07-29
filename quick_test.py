#!/usr/bin/env python3
# Quick test to verify the memorization metrics fix

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
import torch.nn as nn
from memorization_metrics import calculate_model_conditional_entropy
from data_generation import generate_random_binary_sequences

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

def main():
    print("Testing memorization metrics fix...")
    
    # Create test dataset
    dataset = generate_random_binary_sequences(4, 8, 2, 42)
    print(f"Dataset shape: {dataset.shape}")
    print(f"Sample data: {dataset[0]}")
    
    # Create perfect memory model
    model = SimplePerfectMemoryModel(vocab_size=2, seq_length=8)
    
    # Test conditional entropy calculation
    device = 'cpu'
    conditional_entropy = calculate_model_conditional_entropy(model, dataset, device, batch_size=2)
    
    print(f"Conditional entropy: {conditional_entropy}")
    print(f"Expected: Very small value < 0.01")
    
    if conditional_entropy < 0.01:
        print("✅ TEST PASSED: Perfect memory model has very low conditional entropy")
        return True
    else:
        print("❌ TEST FAILED: Conditional entropy too high for perfect memory model")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

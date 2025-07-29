#!/usr/bin/env python3
# run_improved_morris.py
# Run Morris validation with better parameters for clearer results

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from training_loop import run_morris_validation_experiment, quick_morris_test

def run_improved_morris_experiment():
    """Run Morris validation with improved parameters to see scaling law."""
    print("🎯 Improved Morris Validation Experiment")
    print("=" * 60)
    print("Using larger models and longer training to see Morris scaling law more clearly")
    
    # Improved configuration for better memorization
    improved_config = {
        'max_steps': 2000,          # More training steps
        'learning_rate': 1e-3,      # Higher learning rate
        'batch_size': 16,           # Smaller batches for better memorization
        'eval_interval': 200,       # Less frequent evaluation
        'memorization_eval_interval': 400,  # Less frequent memorization evaluation
        'warmup_steps': 200,        # Longer warmup
        'weight_decay': 0.001,      # Lower weight decay for more memorization
        'grad_clip_norm': 0.5       # Lower gradient clipping
    }
    
    print(f"🔧 Improved Configuration:")
    print(f"   Training steps: {improved_config['max_steps']}")
    print(f"   Learning rate: {improved_config['learning_rate']}")
    print(f"   Batch size: {improved_config['batch_size']}")
    print(f"   Weight decay: {improved_config['weight_decay']} (lower = more memorization)")
    
    # Run with larger models and datasets
    results = run_morris_validation_experiment(
        experiment_name="improved_morris_validation",
        model_names=['micro', 'mini'],  # Skip nano, use larger models
        dataset_sizes=[1000, 2000],     # Larger datasets
        base_config_overrides=improved_config,
        log_base_dir=Path("logs_improved"),
        cache_dir=Path("cache_improved"),
        resume_incomplete=True
    )
    
    return results

def run_quick_comparison():
    """Run a quick comparison between nano and micro models."""
    print("\n🔬 Quick Model Comparison")
    print("=" * 40)
    
    models_to_test = ['nano', 'micro']
    results = []
    
    for model_name in models_to_test:
        print(f"\n📊 Testing {model_name.upper()} model...")
        
        result = quick_morris_test(
            model_name=model_name,
            dataset_size=500,      # Larger dataset
            max_steps=1000         # More training
        )
        
        memo = result['final_memorization']
        print(f"   Parameters: {memo['model_parameters']:,}")
        print(f"   Memorization: {memo['morris_memorization_bits']:.2f} bits")
        print(f"   Bits/param: {memo['bits_per_parameter']:.4f}")
        
        results.append({
            'model': model_name,
            'params': memo['model_parameters'],
            'memorization': memo['morris_memorization_bits'],
            'bits_per_param': memo['bits_per_parameter']
        })
    
    # Compare results
    print(f"\n📈 Comparison Results:")
    print(f"   Model scaling factor: {results[1]['params'] / results[0]['params']:.1f}x parameters")
    print(f"   Memorization scaling: {results[1]['memorization'] / max(results[0]['memorization'], 0.1):.1f}x memorization")
    
    if results[0]['memorization'] > 0 and results[1]['memorization'] > 0:
        expected_scaling = results[1]['params'] / results[0]['params']
        actual_scaling = results[1]['memorization'] / results[0]['memorization']
        print(f"   Expected linear scaling: {expected_scaling:.1f}x")
        print(f"   Actual scaling: {actual_scaling:.1f}x")
        print(f"   Scaling match: {'✅ Good' if abs(expected_scaling - actual_scaling) < 2 else '❌ Poor'}")
    
    return results

def main():
    """Main execution."""
    print("🚀 Improved Morris Validation")
    print("=" * 50)
    
    print("Choose experiment:")
    print("1. Quick model comparison (nano vs micro)")
    print("2. Full improved Morris validation (micro + mini)")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            run_quick_comparison()
        elif choice == '2':
            run_improved_morris_experiment()
        elif choice == '3':
            run_quick_comparison()
            run_improved_morris_experiment()
        else:
            print("Invalid choice. Running quick comparison...")
            run_quick_comparison()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

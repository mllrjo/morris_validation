#!/usr/bin/env python3
# minimal_memorization_test.py
# Debug test: Can nano model perfectly memorize 5 fixed sequences?

import sys
from pathlib import Path
import torch
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fixed_memorization import (
    generate_fixed_memorization_dataset,
    create_memorization_training_config
)
from training_loop import train_model_with_memorization_tracking
from memorization_metrics import evaluate_memorization_on_dataset
from logging_utils import setup_logging_directories, generate_experiment_id

def create_minimal_fixed_dataset(n_sequences=5, seq_length=64, seed=42):
    """Create minimal fixed dataset for perfect memorization test."""
    print(f"🎲 Creating minimal fixed dataset: {n_sequences} sequences")
    
    # Generate the sequences
    torch.manual_seed(seed)
    dataset = torch.randint(0, 2, (n_sequences, seq_length), dtype=torch.long)
    
    # Print sequences for verification
    print("📝 Fixed sequences to memorize:")
    for i, seq in enumerate(dataset):
        seq_str = ''.join(str(x.item()) for x in seq[:20])  # First 20 tokens
        print(f"   Seq {i}: {seq_str}... (length {len(seq)})")
    
    # Create metadata
    metadata = {
        'generation_params': {
            'n_sequences': n_sequences,
            'seq_length': seq_length,
            'vocab_size': 2,
            'seed': seed
        },
        'dataset_properties': {
            'shape': list(dataset.shape),
            'total_tokens': dataset.numel(),
            'theoretical_entropy': dataset.numel(),  # 1 bit per token
        },
        'memorization_info': {
            'dataset_type': 'fixed_memorization',
            'unique_sequences': n_sequences,
            'theoretical_max_memorization': n_sequences * seq_length
        }
    }
    
    return dataset, metadata

def run_minimal_memorization_test():
    """Run minimal test: 5 sequences until perfect memorization."""
    print("🧪 Minimal Memorization Test")
    print("=" * 50)
    print("Goal: Perfect memorization of 5 fixed sequences")
    print("Expected: >95% memorization, loss near 0")
    
    # Minimal dataset
    n_sequences = 5
    seq_length = 64
    
    print(f"\n📊 Test Parameters:")
    print(f"   Sequences: {n_sequences}")
    print(f"   Length: {seq_length} tokens each")
    print(f"   Total bits to memorize: {n_sequences * seq_length}")
    print(f"   Nano model params: 27,072")
    print(f"   Target bits/param: {(n_sequences * seq_length) / 27072:.4f}")
    
    # Create dataset
    train_dataset, train_metadata = create_minimal_fixed_dataset(n_sequences, seq_length)
    
    # Use same dataset for evaluation (should get perfect memorization)
    eval_dataset = train_dataset.clone()
    eval_metadata = train_metadata.copy()
    
    # Configuration for convergence
    config = create_memorization_training_config(
        model_name='nano',
        n_sequences=n_sequences,
        epochs=500,  # Many epochs to ensure convergence
        learning_rate=0.003,  # Higher LR for small dataset
        batch_size=5,  # Batch size = dataset size
        eval_interval=100,
        memorization_eval_interval=200
    )
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: 500")
    print(f"   Steps: {config['training']['max_steps']:,}")
    print(f"   Batch size: 5 (full dataset)")
    print(f"   Learning rate: 0.003")
    print(f"   Shuffle: {True}")  # Verify DataLoader shuffles
    
    # Setup logging
    log_dirs = setup_logging_directories(Path("logs_minimal_test"))
    experiment_id = generate_experiment_id()
    
    print(f"\n🎯 Starting minimal memorization training...")
    print(f"   Experiment ID: {experiment_id}")
    
    # Run training
    start_time = time.time()
    
    try:
        results = train_model_with_memorization_tracking(
            config=config,
            experiment_id=experiment_id,
            log_dirs=log_dirs,
            train_dataset=train_dataset,
            train_metadata=train_metadata,
            eval_dataset=eval_dataset,
            eval_metadata=eval_metadata
        )
        
        training_time = time.time() - start_time
        
        # Detailed analysis
        print(f"\n🎉 Training completed in {training_time/60:.1f} minutes")
        print("=" * 50)
        
        final_memo = results['final_memorization']
        
        print(f"📊 Final Results:")
        print(f"   Morris memorization: {final_memo['morris_memorization_bits']:.1f} bits")
        print(f"   Theoretical entropy: {final_memo['theoretical_entropy_bits']:.1f} bits")  
        print(f"   Conditional entropy: {final_memo['conditional_entropy_bits']:.1f} bits")
        print(f"   Bits per parameter: {final_memo['bits_per_parameter']:.4f}")
        print(f"   Memorization fraction: {final_memo['memorization_fraction']:.1%}")
        
        # Diagnostics
        theoretical_max = n_sequences * seq_length
        actual_memorized = final_memo['morris_memorization_bits']
        efficiency = actual_memorized / theoretical_max
        
        print(f"\n🔍 Diagnostics:")
        print(f"   Theoretical maximum: {theoretical_max} bits")
        print(f"   Actually memorized: {actual_memorized:.1f} bits")
        print(f"   Efficiency: {efficiency:.1%}")
        
        # Verdict
        if efficiency > 0.95:
            print(f"\n✅ SUCCESS: Near-perfect memorization achieved!")
            print(f"   The model CAN memorize fixed sequences perfectly.")
            print(f"   Previous 80% result was due to insufficient training or dataset size.")
        elif efficiency > 0.80:
            print(f"\n⚠️  PARTIAL: Good but not perfect memorization")
            print(f"   May need longer training or there's a fundamental limit.")
        else:
            print(f"\n❌ ISSUE: Poor memorization on minimal dataset")
            print(f"   Suggests code error or fundamental problem.")
        
        # Check if we should test with even smaller dataset
        if efficiency < 0.95:
            print(f"\n💡 Suggestion: Try 2-3 sequences for debugging")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_perfect_memorization_threshold():
    """Test progressively smaller datasets to find perfect memorization threshold."""
    print("\n🔬 Perfect Memorization Threshold Test")
    print("=" * 50)
    
    # Test with 1, 2, 3, 5 sequences
    sequence_counts = [1, 2, 3, 5]
    results = []
    
    for n_seq in sequence_counts:
        print(f"\n📊 Testing {n_seq} sequences...")
        
        # Quick training config
        train_dataset, train_metadata = create_minimal_fixed_dataset(n_seq, 32)  # Shorter sequences
        eval_dataset = train_dataset.clone()
        eval_metadata = train_metadata.copy()
        
        config = create_memorization_training_config(
            model_name='nano',
            n_sequences=n_seq,
            epochs=200,
            learning_rate=0.005,
            batch_size=max(1, n_seq),
            eval_interval=50,
            memorization_eval_interval=100
        )
        
        log_dirs = setup_logging_directories(Path(f"logs_threshold_{n_seq}seq"))
        experiment_id = generate_experiment_id()
        
        try:
            result = train_model_with_memorization_tracking(
                config=config,
                experiment_id=experiment_id,
                log_dirs=log_dirs,
                train_dataset=train_dataset,
                train_metadata=train_metadata,
                eval_dataset=eval_dataset,
                eval_metadata=eval_metadata
            )
            
            efficiency = result['final_memorization']['memorization_fraction']
            bits_per_param = result['final_memorization']['bits_per_parameter']
            
            print(f"   {n_seq} sequences: {efficiency:.1%} memorization, {bits_per_param:.4f} bits/param")
            
            results.append({
                'n_sequences': n_seq,
                'efficiency': efficiency,
                'bits_per_param': bits_per_param
            })
            
        except Exception as e:
            print(f"   {n_seq} sequences: FAILED ({e})")
    
    # Analysis
    print(f"\n📈 Threshold Analysis:")
    for result in results:
        n_seq = result['n_sequences']
        eff = result['efficiency']
        status = "✅ PERFECT" if eff > 0.95 else "⚠️ PARTIAL" if eff > 0.8 else "❌ POOR"
        print(f"   {n_seq} sequences: {eff:.1%} - {status}")
    
    return results

def main():
    """Main test controller."""
    print("🔍 Minimal Memorization Debug Test")
    print("=" * 50)
    print("Purpose: Debug the 80% memorization issue")
    print("Hypothesis: Model should achieve near-perfect memorization on small fixed datasets")
    
    print("\nTest options:")
    print("1. Minimal test: 5 sequences (recommended)")
    print("2. Threshold test: 1,2,3,5 sequences")
    print("3. Both tests")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            run_minimal_memorization_test()
        elif choice == '2':
            test_perfect_memorization_threshold()
        elif choice == '3':
            run_minimal_memorization_test()
            test_perfect_memorization_threshold()
        else:
            print("Running minimal test...")
            run_minimal_memorization_test()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

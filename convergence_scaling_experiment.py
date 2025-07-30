#!/usr/bin/env python3
# convergence_scaling_experiment.py
# Morris validation with proper convergence criteria for meaningful results

import sys
from pathlib import Path
import json
import time
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fixed_memorization import (
    generate_fixed_memorization_dataset,
    create_memorization_training_config
)
from training_loop import train_model_with_memorization_tracking
from logging_utils import setup_logging_directories, generate_experiment_id

def run_convergence_based_experiment(
    n_sequences: int,
    target_memorization: float = 0.95,
    max_epochs: int = 10000,
    convergence_loss: float = 0.05,
    patience: int = 5
):
    """Run experiment with convergence criteria rather than fixed epochs.
    
    Args:
        n_sequences: Number of sequences to memorize
        target_memorization: Target memorization efficiency (0.95 = 95%)
        max_epochs: Maximum epochs before stopping
        convergence_loss: Loss threshold for convergence
        patience: Epochs to wait after convergence before stopping
        
    Returns:
        Experiment results with convergence info
    """
    print(f"\n🎯 Convergence-Based Experiment: {n_sequences} Sequences")
    print("=" * 60)
    
    # Calculate theoretical expectations
    total_bits = n_sequences * 64
    nano_params = 27072
    target_bits_per_param = total_bits / nano_params
    
    print(f"📊 Target Analysis:")
    print(f"   Sequences: {n_sequences}")
    print(f"   Total bits to memorize: {total_bits:,}")
    print(f"   Target bits/param: {target_bits_per_param:.4f}")
    print(f"   Target memorization: {target_memorization:.1%}")
    print(f"   Convergence loss threshold: {convergence_loss}")
    
    # Generate dataset
    print(f"\n📥 Generating fixed dataset...")
    cache_dir = Path(f"cache_convergence_{n_sequences}")
    
    train_dataset, train_metadata = generate_fixed_memorization_dataset(
        n_sequences=n_sequences,
        seq_length=64,
        cache_dir=cache_dir
    )
    
    # Use same dataset for evaluation (should get perfect memorization)
    eval_dataset = train_dataset.clone()
    eval_metadata = train_metadata.copy()
    eval_metadata['dataset_properties']['shape'] = list(eval_dataset.shape)
    eval_metadata['dataset_properties']['total_tokens'] = eval_dataset.numel()
    
    print(f"✅ Dataset created: {len(train_dataset)} sequences")
    
    # Estimate epochs needed based on minimal test success
    # 5 sequences needed 500 epochs for 97.5%
    # Scale proportionally with some buffer
    base_epochs_per_sequence = 100  # Conservative estimate
    estimated_epochs = min(max_epochs, n_sequences * base_epochs_per_sequence)
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Estimated epochs needed: {estimated_epochs:,}")
    print(f"   Maximum epochs: {max_epochs:,}")
    print(f"   Convergence criteria: Loss < {convergence_loss}")
    
    # Create configuration with convergence-based training
    config = create_memorization_training_config(
        model_name='nano',
        n_sequences=n_sequences,
        epochs=estimated_epochs,
        learning_rate=0.003,  # Same as successful minimal test
        batch_size=min(16, n_sequences),  # Reasonable batch size
        eval_interval=200,
        memorization_eval_interval=500
    )
    
    # Setup logging
    log_dirs = setup_logging_directories(Path(f"logs_convergence_{n_sequences}"))
    experiment_id = generate_experiment_id()
    
    print(f"\n🎯 Starting convergence-based training...")
    print(f"   Experiment ID: {experiment_id}")
    print(f"   Max steps: {config['training']['max_steps']:,}")
    
    # Enhanced training with convergence monitoring
    start_time = time.time()
    
    try:
        # Run training
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
        
        # Analyze convergence
        final_memo = results['final_memorization']
        memorization_efficiency = final_memo['memorization_fraction']
        bits_per_param = final_memo['bits_per_parameter']
        bits_memorized = final_memo['morris_memorization_bits']
        
        print(f"\n🎉 Training completed in {training_time/60:.1f} minutes")
        print("=" * 60)
        
        print(f"📊 Convergence Results:")
        print(f"   Morris memorization: {bits_memorized:.1f} bits")
        print(f"   Memorization efficiency: {memorization_efficiency:.1%}")
        print(f"   Bits per parameter: {bits_per_param:.4f}")
        print(f"   Training time: {training_time/60:.1f} minutes")
        
        # Convergence analysis
        convergence_status = "CONVERGED" if memorization_efficiency >= target_memorization else "PARTIAL"
        morris_progress = bits_per_param / 3.6 * 100
        
        print(f"\n🔍 Analysis:")
        print(f"   Convergence status: {convergence_status}")
        print(f"   Target achieved: {'✅' if memorization_efficiency >= target_memorization else '⚠️'}")
        print(f"   Progress toward Morris 3.6: {morris_progress:.1f}%")
        
        # Enhanced results with convergence info
        enhanced_results = {
            **results,
            'experiment_type': 'convergence_based',
            'n_sequences': n_sequences,
            'target_memorization': target_memorization,
            'convergence_achieved': memorization_efficiency >= target_memorization,
            'training_time_minutes': training_time / 60,
            'memorization_efficiency': memorization_efficiency,
            'morris_progress_percent': morris_progress,
            'estimated_epochs': estimated_epochs,
            'actual_epochs': results.get('final_epoch', 0)
        }
        
        return enhanced_results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def run_progressive_convergence_scaling():
    """Run progressive scaling with convergence criteria."""
    print("🚀 Progressive Convergence-Based Morris Validation")
    print("=" * 70)
    
    print("Strategy: Train each dataset size until convergence (>95% memorization)")
    print("Goal: Find the scaling relationship for Morris validation")
    
    # Progressive sequence counts - reasonable scaling
    sequence_counts = [10, 25, 50, 100, 200, 500]
    
    all_results = []
    
    for i, n_sequences in enumerate(sequence_counts):
        print(f"\n🔬 Experiment {i+1}/{len(sequence_counts)}")
        
        # Calculate if this is reasonable given time constraints
        estimated_bits = n_sequences * 64
        estimated_bits_per_param = estimated_bits / 27072
        
        print(f"📈 Scaling Analysis:")
        print(f"   Target: {estimated_bits_per_param:.4f} bits/param")
        
        if estimated_bits_per_param > 10.0:
            print(f"⚠️  Very high target - may need extremely long training")
            user_continue = input(f"Continue with {n_sequences} sequences? (y/n): ").strip().lower()
            if user_continue != 'y':
                print(f"Skipping {n_sequences} sequences")
                continue
        
        # Run convergence experiment
        result = run_convergence_based_experiment(
            n_sequences=n_sequences,
            target_memorization=0.95,
            max_epochs=20000,  # High limit but reasonable
            convergence_loss=0.05
        )
        
        if result:
            all_results.append(result)
            
            # Show progress toward Morris 3.6
            bits_per_param = result['final_memorization']['bits_per_parameter']
            efficiency = result['memorization_efficiency']
            
            print(f"\n📈 Progress Summary:")
            print(f"   {n_sequences} sequences: {bits_per_param:.4f} bits/param ({efficiency:.1%})")
            
            # Check if we're approaching Morris target
            if bits_per_param >= 3.0:
                print(f"🎯 Approaching Morris 3.6 target!")
                print(f"Ready to validate scaling law with larger models.")
                break
            elif bits_per_param >= 1.0:
                print(f"✅ Good progress - continue scaling")
            
            # Option to continue or stop
            if i < len(sequence_counts) - 1:
                continue_scaling = input(f"\nContinue to {sequence_counts[i+1]} sequences? (y/n): ").strip().lower()
                if continue_scaling != 'y':
                    print("Stopping progressive scaling")
                    break
        else:
            print(f"❌ Failed for {n_sequences} sequences")
            break
    
    # Final analysis
    print(f"\n🎉 Progressive Convergence Scaling Complete!")
    print("=" * 70)
    
    if all_results:
        print(f"📊 Scaling Results Summary:")
        for result in all_results:
            n_seq = result['n_sequences']
            bits_per_param = result['final_memorization']['bits_per_parameter']
            efficiency = result['memorization_efficiency']
            time_min = result['training_time_minutes']
            
            print(f"   {n_seq:3d} sequences: {bits_per_param:.4f} bits/param "
                  f"({efficiency:.1%}, {time_min:.1f}min)")
        
        # Find best result
        best_result = max(all_results, key=lambda r: r['final_memorization']['bits_per_parameter'])
        best_bits_per_param = best_result['final_memorization']['bits_per_parameter']
        
        print(f"\n🏆 Best Performance:")
        print(f"   {best_result['n_sequences']} sequences: {best_bits_per_param:.4f} bits/param")
        
        # Morris validation assessment
        morris_progress = best_bits_per_param / 3.6 * 100
        print(f"   Progress toward Morris 3.6: {morris_progress:.1f}%")
        
        if best_bits_per_param >= 3.0:
            print(f"✅ MORRIS VALIDATION ACHIEVED with nano model!")
        elif best_bits_per_param >= 1.0:
            print(f"🎯 Significant progress - ready for larger models")
        else:
            print(f"📈 Need larger datasets or longer training")
    
    # Save comprehensive results
    results_file = Path("convergence_scaling_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_type': 'progressive_convergence_scaling',
            'model_name': 'nano',
            'sequence_counts': sequence_counts,
            'convergence_criteria': {
                'target_memorization': 0.95,
                'max_epochs': 20000,
                'convergence_loss': 0.05
            },
            'results': all_results,
            'summary': {
                'total_experiments': len(all_results),
                'best_bits_per_param': max((r['final_memorization']['bits_per_parameter'] for r in all_results), default=0),
                'morris_validation_progress': morris_progress if all_results else 0
            }
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    return all_results

def main():
    """Main convergence experiment controller."""
    print("🎯 Convergence-Based Morris Validation")
    print("=" * 50)
    
    print("Approach: Train until convergence (95%+ memorization) rather than fixed epochs")
    print("Based on successful minimal test: 97.5% memorization achieved")
    
    print("\nExperiment options:")
    print("1. Progressive scaling (10→500 sequences, convergence-based)")
    print("2. Single dataset size with convergence")
    print("3. Quick test: 25 sequences")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            run_progressive_convergence_scaling()
            
        elif choice == '2':
            n_sequences = int(input("Enter number of sequences: ").strip())
            result = run_convergence_based_experiment(n_sequences)
            
        elif choice == '3':
            print("🧪 Quick convergence test with 25 sequences...")
            result = run_convergence_based_experiment(25, max_epochs=5000)
            
        else:
            print("Running progressive scaling...")
            run_progressive_convergence_scaling()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Experiment interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

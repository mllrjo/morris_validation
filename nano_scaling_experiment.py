#!/usr/bin/env python3
# nano_scaling_experiment.py
# Systematic scaling experiments for nano model toward Morris 3.6 validation

import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fixed_memorization import (
    generate_fixed_memorization_dataset,
    create_memorization_training_config,
    estimate_memorization_capacity_for_model
)
from training_loop import train_model_with_memorization_tracking
from logging_utils import setup_logging_directories, generate_experiment_id

def run_nano_dataset_scaling_experiment():
    """Phase 1: Scale dataset size with nano model."""
    print("🎯 Phase 1: Nano Dataset Scaling Experiment")
    print("=" * 60)
    
    # Progressive dataset sizes
    dataset_sizes = [200, 500, 1000, 1500, 2000]
    model_name = 'nano'
    epochs = 150  # Extended training
    
    results = []
    
    for i, n_sequences in enumerate(dataset_sizes):
        print(f"\n📊 Dataset Scale Test {i+1}/{len(dataset_sizes)}")
        print(f"Sequences: {n_sequences}, Target: {n_sequences * 64:,} bits")
        
        # Capacity analysis
        capacity = estimate_memorization_capacity_for_model(model_name, n_sequences, 64)
        print(f"Capacity: {capacity['recommendation']}")
        
        # Estimate target bits/param
        model_params = 27072  # nano model
        expected_bits = n_sequences * 64  # Total bits to memorize
        target_bits_per_param = expected_bits / model_params
        print(f"Target bits/param: {target_bits_per_param:.3f}")
        
        # Generate dataset
        print("📥 Generating fixed dataset...")
        cache_dir = Path(f"cache_nano_scale_{n_sequences}")
        
        train_dataset, train_metadata = generate_fixed_memorization_dataset(
            n_sequences=n_sequences,
            seq_length=64,
            cache_dir=cache_dir
        )
        
        # Evaluation dataset (20% of training)
        eval_size = max(50, n_sequences // 5)
        eval_dataset = train_dataset[:eval_size]
        eval_metadata = train_metadata.copy()
        eval_metadata['dataset_properties']['shape'] = list(eval_dataset.shape)
        eval_metadata['dataset_properties']['total_tokens'] = eval_dataset.numel()
        
        # Create config with extended training
        config = create_memorization_training_config(
            model_name=model_name,
            n_sequences=n_sequences,
            epochs=epochs,
            learning_rate=0.002,
            batch_size=16,
            eval_interval=300,
            memorization_eval_interval=600
        )
        
        # Setup logging
        log_dirs = setup_logging_directories(Path(f"logs_nano_scale_{n_sequences}"))
        experiment_id = generate_experiment_id()
        
        print(f"🎯 Training: {config['training']['max_steps']:,} steps ({epochs} epochs)")
        
        # Run training
        start_time = time.time()
        
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
            
            training_time = time.time() - start_time
            
            # Analyze results
            final_memo = result['final_memorization']
            bits_memorized = final_memo['morris_memorization_bits']
            bits_per_param = final_memo['bits_per_parameter']
            efficiency = final_memo['memorization_fraction']
            
            print(f"\n📊 Results for {n_sequences} sequences:")
            print(f"   Morris memorization: {bits_memorized:.1f} bits")
            print(f"   Bits per parameter: {bits_per_param:.4f}")
            print(f"   Memorization efficiency: {efficiency:.1%}")
            print(f"   Training time: {training_time/60:.1f} minutes")
            
            # Progress toward Morris 3.6
            morris_progress = bits_per_param / 3.6 * 100
            print(f"   Progress toward Morris 3.6: {morris_progress:.1f}%")
            
            # Store result
            results.append({
                'n_sequences': n_sequences,
                'bits_memorized': bits_memorized,
                'bits_per_parameter': bits_per_param,
                'memorization_efficiency': efficiency,
                'training_time_minutes': training_time / 60,
                'target_bits_per_param': target_bits_per_param,
                'morris_progress_percent': morris_progress,
                'experiment_id': experiment_id
            })
            
        except Exception as e:
            print(f"❌ Training failed for {n_sequences} sequences: {e}")
            continue
    
    # Summary analysis
    print(f"\n🎉 Phase 1 Complete: Dataset Scaling Results")
    print("=" * 60)
    
    for result in results:
        n_seq = result['n_sequences']
        bits_per_param = result['bits_per_parameter']
        progress = result['morris_progress_percent']
        efficiency = result['memorization_efficiency']
        
        print(f"   {n_seq:,} sequences: {bits_per_param:.4f} bits/param "
              f"({progress:.1f}% toward 3.6, {efficiency:.1%} efficiency)")
    
    # Find best performance
    if results:
        best_result = max(results, key=lambda r: r['bits_per_parameter'])
        print(f"\n🏆 Best Performance:")
        print(f"   {best_result['n_sequences']:,} sequences: {best_result['bits_per_parameter']:.4f} bits/param")
        
        # Recommendation for Phase 2
        if best_result['bits_per_parameter'] < 1.0:
            print(f"\n📈 Recommendation for Phase 2:")
            print(f"   Use {best_result['n_sequences']:,} sequences with extended training (300+ epochs)")
        else:
            print(f"\n🎯 Ready for Phase 3: Model scaling!")
    
    # Save results
    results_file = Path("nano_dataset_scaling_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_type': 'nano_dataset_scaling',
            'model_name': 'nano',
            'dataset_sizes': dataset_sizes,
            'epochs': epochs,
            'results': results,
            'summary': {
                'total_experiments': len(results),
                'best_bits_per_param': max(r['bits_per_parameter'] for r in results) if results else 0,
                'best_dataset_size': best_result['n_sequences'] if results else None
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    return results

def run_nano_extended_training_experiment(best_dataset_size: int = 1000):
    """Phase 2: Extended training with optimal dataset size."""
    print(f"\n🎯 Phase 2: Nano Extended Training Experiment")
    print("=" * 60)
    
    # Extended epoch configurations
    epoch_configs = [
        {'epochs': 200, 'lr': 0.002},
        {'epochs': 300, 'lr': 0.0015},
        {'epochs': 400, 'lr': 0.001}
    ]
    
    model_name = 'nano'
    n_sequences = best_dataset_size
    
    print(f"Using optimal dataset size: {n_sequences} sequences")
    print(f"Target: {n_sequences * 64:,} bits ({(n_sequences * 64 / 27072):.3f} bits/param)")
    
    results = []
    
    # Generate dataset once
    print("📥 Generating fixed dataset...")
    cache_dir = Path(f"cache_nano_extended_{n_sequences}")
    
    train_dataset, train_metadata = generate_fixed_memorization_dataset(
        n_sequences=n_sequences,
        seq_length=64,
        cache_dir=cache_dir
    )
    
    eval_size = max(100, n_sequences // 5)
    eval_dataset = train_dataset[:eval_size]
    eval_metadata = train_metadata.copy()
    eval_metadata['dataset_properties']['shape'] = list(eval_dataset.shape)
    eval_metadata['dataset_properties']['total_tokens'] = eval_dataset.numel()
    
    for i, epoch_config in enumerate(epoch_configs):
        epochs = epoch_config['epochs']
        lr = epoch_config['lr']
        
        print(f"\n📊 Extended Training Test {i+1}/{len(epoch_configs)}")
        print(f"Epochs: {epochs}, Learning rate: {lr}")
        
        # Create config
        config = create_memorization_training_config(
            model_name=model_name,
            n_sequences=n_sequences,
            epochs=epochs,
            learning_rate=lr,
            batch_size=16,
            eval_interval=500,
            memorization_eval_interval=1000
        )
        
        # Setup logging
        log_dirs = setup_logging_directories(Path(f"logs_nano_extended_{epochs}ep"))
        experiment_id = generate_experiment_id()
        
        print(f"🎯 Training: {config['training']['max_steps']:,} steps")
        
        # Run training
        start_time = time.time()
        
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
            
            training_time = time.time() - start_time
            
            # Analyze results
            final_memo = result['final_memorization']
            bits_memorized = final_memo['morris_memorization_bits']
            bits_per_param = final_memo['bits_per_parameter']
            efficiency = final_memo['memorization_fraction']
            
            print(f"\n📊 Results for {epochs} epochs:")
            print(f"   Morris memorization: {bits_memorized:.1f} bits")
            print(f"   Bits per parameter: {bits_per_param:.4f}")
            print(f"   Memorization efficiency: {efficiency:.1%}")
            print(f"   Training time: {training_time/60:.1f} minutes")
            
            # Progress toward Morris 3.6
            morris_progress = bits_per_param / 3.6 * 100
            print(f"   Progress toward Morris 3.6: {morris_progress:.1f}%")
            
            # Store result
            results.append({
                'epochs': epochs,
                'learning_rate': lr,
                'bits_memorized': bits_memorized,
                'bits_per_parameter': bits_per_param,
                'memorization_efficiency': efficiency,
                'training_time_minutes': training_time / 60,
                'morris_progress_percent': morris_progress,
                'experiment_id': experiment_id
            })
            
        except Exception as e:
            print(f"❌ Training failed for {epochs} epochs: {e}")
            continue
    
    # Summary analysis
    print(f"\n🎉 Phase 2 Complete: Extended Training Results")
    print("=" * 60)
    
    for result in results:
        epochs = result['epochs']
        bits_per_param = result['bits_per_parameter']
        progress = result['morris_progress_percent']
        efficiency = result['memorization_efficiency']
        time_min = result['training_time_minutes']
        
        print(f"   {epochs} epochs: {bits_per_param:.4f} bits/param "
              f"({progress:.1f}% toward 3.6, {time_min:.1f}min)")
    
    # Find best performance
    if results:
        best_result = max(results, key=lambda r: r['bits_per_parameter'])
        print(f"\n🏆 Best Performance:")
        print(f"   {best_result['epochs']} epochs: {best_result['bits_per_parameter']:.4f} bits/param")
        
        # Recommendation for Phase 3
        if best_result['bits_per_parameter'] >= 1.0:
            print(f"\n🎯 Ready for Phase 3: Model scaling with {best_result['epochs']} epochs!")
        else:
            print(f"\n📈 Consider even longer training or larger datasets before Phase 3")
    
    # Save results
    results_file = Path("nano_extended_training_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_type': 'nano_extended_training',
            'model_name': 'nano',
            'dataset_size': n_sequences,
            'epoch_configs': epoch_configs,
            'results': results,
            'summary': {
                'total_experiments': len(results),
                'best_bits_per_param': max(r['bits_per_parameter'] for r in results) if results else 0,
                'best_epochs': best_result['epochs'] if results else None
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    return results

def main():
    """Main scaling experiment controller."""
    print("🚀 Systematic Nano Model Scaling for Morris 3.6 Validation")
    print("=" * 70)
    
    print("\nPhased scaling approach:")
    print("Phase 1: Scale dataset size (200 → 2000 sequences)")
    print("Phase 2: Extend training (200 → 400 epochs)")  
    print("Phase 3: Scale model size (nano → micro → mini)")
    
    print("\nChoose experiment phase:")
    print("1. Phase 1: Dataset scaling (recommended first)")
    print("2. Phase 2: Extended training") 
    print("3. Run both phases sequentially")
    print("4. Quick test: 500 sequences, 200 epochs")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            results = run_nano_dataset_scaling_experiment()
            
        elif choice == '2':
            # Use default or ask for dataset size
            dataset_size = input("Enter dataset size for extended training (default 1000): ").strip()
            dataset_size = int(dataset_size) if dataset_size else 1000
            results = run_nano_extended_training_experiment(dataset_size)
            
        elif choice == '3':
            print("🔄 Running both phases sequentially...")
            
            # Phase 1
            phase1_results = run_nano_dataset_scaling_experiment()
            
            if phase1_results:
                # Find best dataset size from Phase 1
                best_result = max(phase1_results, key=lambda r: r['bits_per_parameter'])
                best_dataset_size = best_result['n_sequences']
                
                print(f"\n🔄 Moving to Phase 2 with optimal dataset size: {best_dataset_size}")
                
                # Phase 2
                phase2_results = run_nano_extended_training_experiment(best_dataset_size)
                
                print(f"\n🎉 Both phases complete!")
            
        elif choice == '4':
            print("🧪 Quick scaling test...")
            # Quick test with moderate parameters
            # Implementation similar to dataset scaling but with single config
            pass
            
        else:
            print("Invalid choice. Running Phase 1...")
            run_nano_dataset_scaling_experiment()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Experiment interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

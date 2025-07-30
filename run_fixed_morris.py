#!/usr/bin/env python3
# run_fixed_morris.py - Minimal version for testing
# Morris validation with fixed dataset memorization

import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from fixed_memorization import (
        generate_fixed_memorization_dataset,
        create_memorization_training_config,
        estimate_memorization_capacity_for_model,
        suggest_optimal_memorization_experiments,
        print_memorization_experiment_plan
    )
    from training_loop import train_model_with_memorization_tracking
    from model_architecture import get_morris_model_configs
    from logging_utils import setup_logging_directories, generate_experiment_id
    from memorization_metrics import (
        evaluate_memorization_on_dataset,
        create_memorization_report
    )
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure:")
    print("1. fixed_memorization.py is in src/ directory")
    print("2. training_loop.py has been updated with fixed memorization support")
    sys.exit(1)

def run_quick_nano_test():
    """Run a simple nano model test with fixed memorization."""
    print("🧪 Quick Nano Fixed Memorization Test")
    print("=" * 50)
    
    model_name = 'nano'
    n_sequences = 200
    epochs = 100
    
    print(f"Model: {model_name}")
    print(f"Sequences to memorize: {n_sequences}")
    print(f"Training epochs: {epochs}")
    
    # Show capacity analysis
    try:
        capacity = estimate_memorization_capacity_for_model(model_name, n_sequences, 64)
        print(f"Capacity analysis: {capacity['recommendation']}")
    except Exception as e:
        print(f"Note: Capacity analysis unavailable ({e})")
    
    # Generate fixed dataset
    print("\n📥 Generating fixed dataset...")
    cache_dir = Path(f"cache_fixed_{model_name}")
    
    try:
        train_dataset, train_metadata = generate_fixed_memorization_dataset(
            n_sequences=n_sequences,
            seq_length=64,
            cache_dir=cache_dir
        )
        
        # Create evaluation set (subset of training data)
        eval_size = min(50, n_sequences // 4)
        eval_dataset = train_dataset[:eval_size]
        eval_metadata = train_metadata.copy()
        eval_metadata['dataset_properties']['shape'] = list(eval_dataset.shape)
        eval_metadata['dataset_properties']['total_tokens'] = eval_dataset.numel()
        
        print(f"✅ Dataset created: {len(train_dataset)} training, {len(eval_dataset)} eval sequences")
        
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        return None
    
    # Create config
    print("\n⚙️  Creating training configuration...")
    try:
        config = create_memorization_training_config(
            model_name=model_name,
            n_sequences=n_sequences,
            epochs=epochs,
            learning_rate=0.002,
            batch_size=16
        )
        print(f"✅ Config created: {config['training']['max_steps']} steps")
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return None
    
    # Setup logging
    log_dirs = setup_logging_directories(Path(f"logs_fixed_{model_name}"))
    experiment_id = generate_experiment_id()
    
    print(f"\n🎯 Starting training (ID: {experiment_id})")
    
    # Run training
    try:
        start_time = time.time()
        
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
        
        # Display results
        print(f"\n🎉 Training completed in {training_time/60:.1f} minutes!")
        print("=" * 50)
        
        final_memo = results['final_memorization']
        bits_memorized = final_memo['morris_memorization_bits']
        bits_per_param = final_memo['bits_per_parameter']
        memorization_fraction = final_memo['memorization_fraction']
        
        print(f"📊 Results:")
        print(f"   Morris memorization: {bits_memorized:.1f} bits")
        print(f"   Bits per parameter: {bits_per_param:.4f}")
        print(f"   Memorization fraction: {memorization_fraction:.3f}")
        
        # Compare to theoretical maximum
        theoretical_max = n_sequences * 64
        efficiency = bits_memorized / theoretical_max if theoretical_max > 0 else 0
        print(f"   Memorization efficiency: {efficiency:.1%}")
        
        # Progress toward Morris 3.6
        model_params = final_memo['model_parameters']
        morris_target = model_params * 3.6
        progress = bits_memorized / morris_target if morris_target > 0 else 0
        print(f"   Progress toward Morris 3.6: {progress:.1%}")
        
        print(f"\n💡 Expected improvement over random data: ~50-200x!")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def show_experiment_plan():
    """Show the fixed memorization experiment plan."""
    try:
        print_memorization_experiment_plan()
    except Exception as e:
        print(f"❌ Cannot show experiment plan: {e}")
        print("\nFixed memorization approach:")
        print("• Generate specific sequences once") 
        print("• Train repeatedly on same sequences")
        print("• Model can actually memorize specific patterns")
        print("• Morris H(X) - H(X|θ̂) becomes meaningful")

def main():
    """Main entry point."""
    print("🎯 Fixed Dataset Morris Memorization")
    print("=" * 50)
    
    print("\nOptions:")
    print("1. Quick nano test (recommended)")
    print("2. Show experiment plan")
    print("3. Test imports only")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            run_quick_nano_test()
        elif choice == '2':
            show_experiment_plan()
        elif choice == '3':
            print("✅ All imports working correctly!")
        else:
            print("Running quick nano test...")
            run_quick_nano_test()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

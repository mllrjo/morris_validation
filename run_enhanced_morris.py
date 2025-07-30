#!/usr/bin/env python3
# run_enhanced_morris.py
# Enhanced Morris validation experiment optimized for clear 3.6 bits/parameter results

import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from training_loop import (
    create_training_config,
    run_morris_validation_experiment,
    analyze_morris_scaling_law
)
from model_architecture import get_morris_model_configs
from logging_utils import setup_logging_directories, generate_experiment_id

def create_enhanced_config_overrides():
    """Create optimized configuration for clear Morris scaling results."""
    return {
        # Extended training for significant memorization
        'max_steps': 5000,              # 5x longer than original
        'eval_interval': 200,           # Less frequent loss logging
        'memorization_eval_interval': 500,  # Track memorization every 500 steps
        'save_checkpoint_interval': 1000,   # Save checkpoints every 1000 steps
        
        # Optimized learning for memorization
        'learning_rate': 0.002,         # 4x higher for faster convergence
        'warmup_steps': 200,            # Longer warmup for stability
        'weight_decay': 0.005,          # Reduced regularization
        'grad_clip_norm': 0.5,          # Tighter gradient clipping
        
        # Batch size optimization
        'batch_size': 16,               # Smaller batches for better memorization
        
        # Evaluation dataset size
        'eval_dataset_size': 500        # 5x larger evaluation set
    }

def get_optimal_dataset_sizes():
    """Get dataset sizes optimized for each model's capacity."""
    return {
        'nano': [1000, 2000, 4000],      # Good for small model
        'micro': [3000, 6000, 10000],    # Larger datasets for bigger model
        'mini': [8000, 15000, 25000],    # Much larger for mini model
        'small': [20000, 40000, 60000]   # Massive datasets for small model
    }

def run_focused_model_experiment(model_name: str, dataset_sizes: list, 
                                experiment_name: str = "enhanced_morris"):
    """Run focused experiment on single model with multiple dataset sizes."""
    print(f"\n🎯 Enhanced Morris Experiment: {model_name.upper()} Model")
    print("=" * 60)
    
    # Get model info
    model_configs = get_morris_model_configs()
    if model_name not in model_configs:
        print(f"❌ Unknown model: {model_name}")
        return None
    
    model_info = model_configs[model_name]
    params = model_info['parameters']['total_params']
    print(f"📊 Model: {params:,} parameters")
    print(f"📈 Dataset sizes: {dataset_sizes}")
    print(f"🎯 Target: ~{3.6 * params / 1000:.1f}k bits memorization")
    
    # Enhanced configuration
    config_overrides = create_enhanced_config_overrides()
    print(f"\n🔧 Enhanced Configuration:")
    print(f"   Training steps: {config_overrides['max_steps']:,}")
    print(f"   Learning rate: {config_overrides['learning_rate']}")
    print(f"   Batch size: {config_overrides['batch_size']}")
    print(f"   Eval dataset: {config_overrides['eval_dataset_size']}")
    
    # Run experiment
    start_time = time.time()
    results = run_morris_validation_experiment(
        experiment_name=f"{experiment_name}_{model_name}",
        model_names=[model_name],
        dataset_sizes=dataset_sizes,
        base_config_overrides=config_overrides,
        log_base_dir=Path(f"logs_enhanced_{model_name}"),
        cache_dir=Path(f"cache_enhanced_{model_name}"),
        resume_incomplete=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Experiment completed in {elapsed_time/60:.1f} minutes")
    
    # Analyze results
    if 'individual_results' in results and results['individual_results']:
        print(f"\n📊 {model_name.upper()} Results:")
        print("-" * 40)
        
        for result in results['individual_results']:
            memo = result['final_memorization']
            dataset_size = result['dataset_size']
            bits_per_param = memo['bits_per_parameter']
            total_memo = memo['morris_memorization_bits']
            
            print(f"   Dataset {dataset_size:,}: {total_memo:.1f} bits "
                  f"({bits_per_param:.4f} bits/param)")
        
        # Morris validation analysis
        if 'morris_validation' in results['analysis_report']:
            morris_analysis = results['analysis_report']['morris_validation']
            validation = morris_analysis['scaling_law_validation']
            
            print(f"\n🔬 Morris Scaling Analysis:")
            print(f"   Average: {validation['average_bits_per_param']:.4f} bits/param")
            print(f"   Target: 3.6 bits/param")
            print(f"   Progress: {validation['average_bits_per_param']/3.6*100:.1f}% toward target")
            print(f"   Validation: {'✅ PASSED' if validation['passes_validation'] else '⚠️  IN PROGRESS'}")
        
        # Save model-specific results
        results_file = Path(f"enhanced_morris_{model_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {results_file}")
    
    return results

def run_progressive_validation():
    """Run progressive validation starting with nano, then scaling up."""
    print("🚀 Progressive Enhanced Morris Validation")
    print("=" * 70)
    
    optimal_sizes = get_optimal_dataset_sizes()
    all_results = {}
    
    # Start with nano (fastest)
    print("\n🎯 Phase 1: Nano Model Validation")
    nano_results = run_focused_model_experiment(
        'nano', 
        optimal_sizes['nano'], 
        "progressive_nano"
    )
    all_results['nano'] = nano_results
    
    # Check if nano shows promising results before continuing
    if nano_results and 'individual_results' in nano_results:
        nano_bits_per_param = []
        for result in nano_results['individual_results']:
            bits_per_param = result['final_memorization']['bits_per_parameter']
            nano_bits_per_param.append(bits_per_param)
        
        avg_nano = sum(nano_bits_per_param) / len(nano_bits_per_param)
        print(f"\n📈 Nano average: {avg_nano:.4f} bits/param")
        
        if avg_nano > 0.01:  # If nano shows good memorization
            print("\n✅ Nano showing good memorization, proceeding to micro...")
            
            # Micro model
            print("\n🎯 Phase 2: Micro Model Validation")
            micro_results = run_focused_model_experiment(
                'micro', 
                optimal_sizes['micro'], 
                "progressive_micro"
            )
            all_results['micro'] = micro_results
            
            # Mini model (if micro is promising)
            if micro_results and 'individual_results' in micro_results:
                micro_avg = sum(r['final_memorization']['bits_per_parameter'] 
                               for r in micro_results['individual_results']) / len(micro_results['individual_results'])
                
                if micro_avg > 0.005:  # If micro shows reasonable memorization
                    print("\n🎯 Phase 3: Mini Model Validation")
                    mini_results = run_focused_model_experiment(
                        'mini', 
                        optimal_sizes['mini'], 
                        "progressive_mini"
                    )
                    all_results['mini'] = mini_results
    
    # Combined analysis
    print(f"\n🎉 Progressive Validation Complete!")
    combined_results_file = Path("progressive_morris_validation.json")
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"📊 Combined results: {combined_results_file}")
    
    return all_results

def run_single_model_deep_dive(model_name: str = 'nano'):
    """Run deep dive on single model for maximum memorization."""
    print(f"🔬 Deep Dive: {model_name.upper()} Model Morris Validation")
    print("=" * 60)
    
    # Ultra-enhanced configuration for deep memorization
    deep_config = {
        'max_steps': 10000,             # Very long training
        'eval_interval': 500,
        'memorization_eval_interval': 1000,
        'save_checkpoint_interval': 2000,
        'learning_rate': 0.003,         # High learning rate
        'warmup_steps': 500,            # Long warmup
        'weight_decay': 0.001,          # Minimal regularization
        'grad_clip_norm': 0.3,          # Loose clipping
        'batch_size': 8,                # Very small batches
        'eval_dataset_size': 1000       # Large evaluation set
    }
    
    # Optimal dataset sizes for the model
    optimal_sizes = get_optimal_dataset_sizes()
    dataset_sizes = optimal_sizes.get(model_name, [1000, 2000, 4000])
    
    print(f"⚙️  Deep Configuration:")
    print(f"   Steps: {deep_config['max_steps']:,}")
    print(f"   Learning rate: {deep_config['learning_rate']}")
    print(f"   Batch size: {deep_config['batch_size']}")
    
    results = run_focused_model_experiment(
        model_name, 
        dataset_sizes, 
        f"deep_dive_{model_name}"
    )
    
    return results

def main():
    """Main enhanced Morris validation execution."""
    print("🚀 Enhanced Morris Validation Experiments")
    print("=" * 70)
    
    print("\nChoose experiment type:")
    print("1. Single model deep dive (nano - recommended first)")
    print("2. Single model deep dive (micro)")
    print("3. Single model deep dive (mini)")
    print("4. Progressive validation (all models)")
    print("5. Focused nano experiment (fast validation)")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            results = run_single_model_deep_dive('nano')
        elif choice == '2':
            results = run_single_model_deep_dive('micro')
        elif choice == '3':
            results = run_single_model_deep_dive('mini')
        elif choice == '4':
            results = run_progressive_validation()
        elif choice == '5':
            results = run_focused_model_experiment('nano', [2000, 4000], "focused_nano")
        else:
            print("Invalid choice. Running focused nano experiment...")
            results = run_focused_model_experiment('nano', [2000, 4000], "focused_nano")
        
        print(f"\n🎉 Enhanced Morris Validation Complete!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Experiment interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# example_morris_experiment.py
# Complete example of running Morris validation experiments

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from training_loop import (
    create_training_config,
    run_morris_validation_experiment,
    quick_morris_test,
    analyze_morris_scaling_law,
    validate_morris_scaling_law
)
from model_architecture import get_morris_model_configs
import json

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("🧪 Running Quick Morris Test")
    print("=" * 40)
    
    # Quick test with minimal settings
    results = quick_morris_test(
        model_name='nano',
        dataset_size=50,
        max_steps=100
    )
    
    print("\n📊 Quick Test Results:")
    memo = results['final_memorization']
    print(f"   Model parameters: {memo['model_parameters']:,}")
    print(f"   Morris memorization: {memo['morris_memorization_bits']:.2f} bits")
    print(f"   Bits per parameter: {memo['bits_per_parameter']:.4f}")
    print(f"   Memorization fraction: {memo['memorization_fraction']:.3f}")
    
    return results

def run_single_model_experiment():
    """Run experiment on a single model for demonstration."""
    print("\n🎯 Running Single Model Experiment")
    print("=" * 50)
    
    # Configuration for a focused experiment
    config_overrides = {
        'max_steps': 500,
        'eval_interval': 50,
        'memorization_eval_interval': 100,
        'batch_size': 16,
        'learning_rate': 1e-3
    }
    
    results = run_morris_validation_experiment(
        experiment_name="single_model_demo",
        model_names=['nano'],  # Just one model
        dataset_sizes=[200, 500],  # Two dataset sizes
        base_config_overrides=config_overrides,
        log_base_dir=Path("logs_demo"),
        cache_dir=Path("cache_demo"),
        resume_incomplete=True
    )
    
    if 'individual_results' in results:
        print(f"\n📈 Experiment Results:")
        for result in results['individual_results']:
            memo = result['final_memorization']
            print(f"   Dataset {result['dataset_size']}: "
                  f"{memo['morris_memorization_bits']:.1f} bits, "
                  f"{memo['bits_per_parameter']:.3f} bits/param")
        
        # Show analysis
        if 'morris_validation' in results['analysis_report']:
            morris_analysis = results['analysis_report']['morris_validation']
            if 'scaling_law_validation' in morris_analysis:
                validation = morris_analysis['scaling_law_validation']
                print(f"\n🔬 Morris Validation:")
                print(f"   Average bits/param: {validation['average_bits_per_param']:.3f}")
                print(f"   Passes validation: {'✅' if validation['passes_validation'] else '❌'}")
    
    return results

def run_full_morris_validation():
    """Run the complete Morris validation experiment."""
    print("\n🚀 Running Full Morris Validation Experiment")
    print("=" * 60)
    
    # Show available models
    model_configs = get_morris_model_configs()
    print("📋 Available Models:")
    for name, config in model_configs.items():
        params = config['parameters']['total_params']
        print(f"   {name}: {params:,} parameters")
    
    # Full experiment configuration
    config_overrides = {
        'max_steps': 1000,
        'eval_interval': 100,
        'memorization_eval_interval': 200,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'grad_clip_norm': 1.0
    }
    
    print(f"\n🔧 Experiment Configuration:")
    print(f"   Max steps: {config_overrides['max_steps']}")
    print(f"   Learning rate: {config_overrides['learning_rate']}")
    print(f"   Batch size: {config_overrides['batch_size']}")
    
    # Run comprehensive experiment
    results = run_morris_validation_experiment(
        experiment_name="complete_morris_validation",
        model_names=['nano', 'micro', 'mini'],  # Multiple models
        dataset_sizes=[500, 1000, 2000],  # Multiple dataset sizes
        base_config_overrides=config_overrides,
        log_base_dir=Path("logs_full"),
        cache_dir=Path("cache_full"),
        resume_incomplete=True
    )
    
    if 'individual_results' in results:
        print(f"\n🎉 Full Experiment Completed!")
        print(f"   Total experiments: {len(results['individual_results'])}")
        print(f"   Successful: {results['summary']['successful_experiments']}")
        
        # Detailed results
        print(f"\n📊 Detailed Results:")
        for result in results['individual_results']:
            memo = result['final_memorization']
            print(f"   {result['model_name']} (dataset {result['dataset_size']}): "
                  f"{memo['morris_memorization_bits']:.1f} bits, "
                  f"{memo['bits_per_parameter']:.3f} bits/param")
        
        # Morris scaling analysis
        if 'morris_validation' in results['analysis_report']:
            morris_analysis = results['analysis_report']['morris_validation']
            validation = morris_analysis['scaling_law_validation']
            
            print(f"\n🔬 Morris Scaling Law Validation:")
            print(f"   Target: 3.6 bits/parameter")
            print(f"   Achieved: {validation['average_bits_per_param']:.3f} ± {validation['std_bits_per_param']:.3f}")
            print(f"   Scaling exponent: {validation['scaling_exponent']:.3f} (expected: ~1.0)")
            print(f"   R-squared: {validation['r_squared']:.3f}")
            print(f"   Validation: {'✅ PASSED' if validation['passes_validation'] else '❌ FAILED'}")
        
        # Save results to file
        results_file = Path("morris_validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {results_file}")
    
    return results

def analyze_existing_results():
    """Analyze results from a previous experiment."""
    results_file = Path("morris_validation_results.json")
    
    if not results_file.exists():
        print(f"❌ No results file found at {results_file}")
        return None
    
    print(f"\n📈 Analyzing Existing Results")
    print("=" * 40)
    
    analysis = validate_morris_scaling_law(results_file)
    
    if 'error' in analysis:
        print(f"❌ Analysis failed: {analysis['error']}")
        return None
    
    validation = analysis['scaling_law_validation']
    print(f"📊 Morris Scaling Law Analysis:")
    print(f"   Data points: {analysis['data_points']}")
    print(f"   Average bits/param: {validation['average_bits_per_param']:.3f}")
    print(f"   Scaling exponent: {validation['scaling_exponent']:.3f}")
    print(f"   R-squared: {validation['r_squared']:.3f}")
    print(f"   Validation: {'✅ PASSED' if validation['passes_validation'] else '❌ FAILED'}")
    
    return analysis

def main():
    """Main example execution."""
    print("🎯 Morris Validation Experiment Examples")
    print("=" * 60)
    
    print("\nChoose an experiment to run:")
    print("1. Quick test (fast, minimal)")
    print("2. Single model experiment (moderate)")  
    print("3. Full Morris validation (comprehensive)")
    print("4. Analyze existing results")
    print("5. Show model configurations")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            run_quick_test()
            
        elif choice == '2':
            run_single_model_experiment()
            
        elif choice == '3':
            run_full_morris_validation()
            
        elif choice == '4':
            analyze_existing_results()
            
        elif choice == '5':
            show_model_configurations()
            
        else:
            print("Invalid choice. Running quick test...")
            run_quick_test()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Experiment interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

def show_model_configurations():
    """Display all available model configurations."""
    print("\n📋 Morris Model Configurations")
    print("=" * 50)
    
    configs = get_morris_model_configs()
    
    for name, config in configs.items():
        arch = config['architecture']
        params = config['parameters']
        memory = config['memory_estimate']
        
        print(f"\n🔹 {name.upper()} Model:")
        print(f"   Layers: {arch['n_layers']}")
        print(f"   Model dim: {arch['d_model']}")
        print(f"   Attention heads: {arch['n_heads']}")
        print(f"   Total parameters: {params['total_params']:,}")
        print(f"   Memory (params): {memory['params_mb']:.1f} MB")
        print(f"   Memory (activations): {memory['activations_mb_per_sample']:.1f} MB/sample")

if __name__ == "__main__":
    main()

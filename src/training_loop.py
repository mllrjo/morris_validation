# src/training_loop.py
# Complete training loop with Morris memorization tracking and fixed memorization support

import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from model_architecture import create_gpt_model, get_morris_model_configs, detect_device, count_model_parameters
from data_generation import get_or_generate_dataset, create_multiple_datasets
from memorization_metrics import (
    evaluate_memorization_on_dataset,
    track_memorization_during_training,
    create_memorization_report,
    memorization_efficiency_score,
    estimate_memorization_capacity
)
from logging_utils import (
    setup_logging_directories,
    generate_experiment_id,
    log_experiment_metadata,
    log_metrics_csv,
    log_training_step
)
from checkpoint_manager import (
    save_experiment_state,
    load_experiment_state,
    save_model_checkpoint,
    load_model_checkpoint,
    mark_experiment_complete,
    get_experiment_progress,
    find_incomplete_experiments
)

def create_training_config(
    model_name: str = 'nano',
    dataset_size: int = 1000,
    seq_length: int = 64,
    vocab_size: int = 2,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_steps: int = 1000,
    eval_interval: int = 100,
    memorization_eval_interval: int = 200,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
    device: Optional[str] = None,
    seed: int = 42,
    save_checkpoint_interval: int = 500,
    eval_dataset_size: int = 100
) -> Dict[str, Any]:
    """Create training configuration dictionary.
    
    Args:
        model_name: Model size ('nano', 'micro', 'mini', 'small')
        dataset_size: Number of training sequences
        seq_length: Sequence length
        vocab_size: Vocabulary size (2 for binary)
        learning_rate: Initial learning rate
        batch_size: Training batch size
        max_steps: Maximum training steps
        eval_interval: Steps between loss evaluations
        memorization_eval_interval: Steps between memorization measurements
        warmup_steps: Learning rate warmup steps
        weight_decay: L2 regularization strength
        grad_clip_norm: Gradient clipping norm
        device: Device to use (auto-detect if None)
        seed: Random seed for reproducibility
        save_checkpoint_interval: Steps between model checkpoints
        eval_dataset_size: Size of evaluation dataset
        
    Returns:
        Complete training configuration dictionary
    """
    if device is None:
        device = detect_device()
    
    # Get model configuration
    model_configs = get_morris_model_configs()
    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_configs.keys())}")
    
    model_config = model_configs[model_name]
    
    config = {
        'model': {
            'name': model_name,
            'n_layers': model_config['architecture']['n_layers'],
            'd_model': model_config['architecture']['d_model'],
            'n_heads': model_config['architecture']['n_heads'],
            'vocab_size': vocab_size,
            'seq_length': seq_length,
            'total_params': model_config['parameters']['total_params']
        },
        'data': {
            'dataset_size': dataset_size,
            'eval_dataset_size': eval_dataset_size,
            'seq_length': seq_length,
            'vocab_size': vocab_size,
            'seed': seed
        },
        'training': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_steps': max_steps,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay,
            'grad_clip_norm': grad_clip_norm,
            'device': device
        },
        'evaluation': {
            'eval_interval': eval_interval,
            'memorization_eval_interval': memorization_eval_interval,
            'save_checkpoint_interval': save_checkpoint_interval
        },
        'experiment': {
            'seed': seed,
            'created_time': time.time()
        }
    }
    
    return config

def create_dataloader(dataset: torch.Tensor, batch_size: int, 
                     shuffle: bool = True, device: str = 'cpu') -> torch.utils.data.DataLoader:
    """Create DataLoader for training dataset.
    
    Args:
        dataset: Dataset tensor of shape (n_sequences, seq_length)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        device: Device for data loading
        
    Returns:
        DataLoader for the dataset
    """
    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, sequences):
            self.sequences = sequences
            
        def __len__(self):
            return len(self.sequences)
            
        def __getitem__(self, idx):
            return self.sequences[idx]
    
    dataset_obj = SequenceDataset(dataset)
    
    # Pin memory only for CUDA, not for MPS or CPU
    use_pin_memory = (device == 'cuda')
    
    return torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=use_pin_memory,
        num_workers=0  # Keep simple for now
    )

def compute_learning_rate(step: int, max_lr: float, warmup_steps: int, 
                         max_steps: int) -> float:
    """Compute learning rate with warmup and cosine decay.
    
    Args:
        step: Current training step
        max_lr: Maximum learning rate
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        
    Returns:
        Current learning rate
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    else:
        # Cosine decay
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        decay_ratio = min(1.0, decay_ratio)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return max_lr * coeff

def train_fixed_memorization_model(
    config: Dict[str, Any],
    experiment_id: str,
    log_dirs: Dict[str, Path],
    fixed_dataset: torch.Tensor,
    dataset_metadata: Dict[str, Any],
    eval_dataset: torch.Tensor,
    eval_metadata: Dict[str, Any],
    resume_from_step: int = 0
) -> Dict[str, Any]:
    """Train model on fixed dataset for memorization experiments.
    
    This function trains a model repeatedly on the same fixed dataset across
    multiple epochs to enable memorization, rather than training on different
    random data each time.
    
    Args:
        config: Training configuration
        experiment_id: Unique experiment identifier
        log_dirs: Logging directory paths
        fixed_dataset: Fixed dataset to memorize (same sequences every epoch)
        dataset_metadata: Dataset metadata
        eval_dataset: Evaluation dataset (subset of fixed_dataset)
        eval_metadata: Evaluation dataset metadata
        resume_from_step: Step to resume from (0 for new training)
        
    Returns:
        Training results and final metrics
    """
    device = config['training']['device']
    
    # Set random seed for reproducibility
    torch.manual_seed(config['experiment']['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(config['experiment']['seed'])
    
    # Create model
    model = create_gpt_model(
        n_layers=config['model']['n_layers'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        vocab_size=config['model']['vocab_size'],
        seq_length=config['model']['seq_length'],
        device=device
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    # Resume from checkpoint if needed
    start_step = resume_from_step
    if start_step > 0:
        loaded_step, success = load_model_checkpoint(
            model, optimizer, experiment_id, log_dirs['checkpoints']
        )
        if success:
            start_step = loaded_step
            print(f"Resumed from checkpoint at step {start_step}")
        else:
            print(f"Failed to load checkpoint, starting from step 0")
            start_step = 0
    
    # Create data loader for fixed dataset
    train_loader = create_dataloader(
        fixed_dataset, 
        config['training']['batch_size'], 
        shuffle=True,  # Shuffle for each epoch
        device=device
    )
    
    # Training state
    model.train()
    running_loss = 0.0
    step = start_step
    current_epoch = start_step // len(train_loader) if len(train_loader) > 0 else 0
    
    # Fixed memorization specific logging
    memorization_history = []
    
    print(f"🎯 Fixed Memorization Training")
    print(f"   Dataset: {len(fixed_dataset)} sequences")
    print(f"   Target epochs: {config['training'].get('epochs', 'N/A')}")
    print(f"   Steps per epoch: {len(train_loader)}")
    print(f"   Total steps: {config['training']['max_steps']:,}")
    print(f"   Starting from step {start_step} (epoch {current_epoch})")
    
    # Training loop
    epoch_start_time = time.time()
    
    while step < config['training']['max_steps']:
        try:
            # Start new epoch
            current_epoch = step // len(train_loader) if len(train_loader) > 0 else 0
            epoch_step = 0
            
            for batch in train_loader:
                if step >= config['training']['max_steps']:
                    break
                
                batch = batch.to(device)
                
                # Compute learning rate
                lr = compute_learning_rate(
                    step, 
                    config['training']['learning_rate'],
                    config['training']['warmup_steps'],
                    config['training']['max_steps']
                )
                
                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Forward pass
                logits = model(batch)
                
                # Compute loss (autoregressive language modeling)
                input_ids = batch[:, :-1]  # All but last token
                target_ids = batch[:, 1:]  # All but first token
                logits = logits[:, :-1, :]  # Match target length
                
                # Flatten for cross-entropy
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if config['training']['grad_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config['training']['grad_clip_norm']
                    )
                
                optimizer.step()
                
                # Update running loss
                running_loss += loss.item()
                step += 1
                epoch_step += 1
                
                # Logging and evaluation
                if step % config['evaluation']['eval_interval'] == 0:
                    avg_loss = running_loss / config['evaluation']['eval_interval']
                    
                    # Calculate epoch progress
                    epoch_progress = epoch_step / len(train_loader) if len(train_loader) > 0 else 0
                    
                    print(f"Step {step:,}/{config['training']['max_steps']:,} "
                          f"(Epoch {current_epoch}.{epoch_progress:.1f}): "
                          f"Loss={avg_loss:.4f}, LR={lr:.6f}")
                    
                    # Log training step
                    log_training_step(experiment_id, step, avg_loss, 0.0, log_dirs['training'])
                    
                    running_loss = 0.0
                
                # Memorization evaluation
                if step % config['evaluation']['memorization_eval_interval'] == 0:
                    print(f"🔍 Evaluating memorization at step {step} (epoch {current_epoch})...")
                    
                    # Switch to eval mode
                    model.eval()
                    
                    with torch.no_grad():
                        # Track memorization on evaluation dataset
                        memo_metrics = track_memorization_during_training(
                            model, eval_dataset, eval_metadata, step,
                            experiment_id, log_dirs['training'], device
                        )
                        
                        # Add epoch information
                        memo_metrics['epoch'] = current_epoch
                        memo_metrics['epoch_progress'] = epoch_step / len(train_loader) if len(train_loader) > 0 else 0
                        memorization_history.append(memo_metrics)
                        
                        bits_memorized = memo_metrics['morris_memorization_bits']
                        bits_per_param = memo_metrics['bits_per_parameter']
                        memo_fraction = memo_metrics['memorization_fraction']
                        
                        print(f"  📊 Morris memorization: {bits_memorized:.1f} bits")
                        print(f"  📊 Bits per parameter: {bits_per_param:.4f}")
                        print(f"  📊 Memorization fraction: {memo_fraction:.3f}")
                        
                        # Show memorization progress
                        if len(memorization_history) > 1:
                            prev_bits = memorization_history[-2]['morris_memorization_bits']
                            improvement = bits_memorized - prev_bits
                            print(f"  📈 Improvement: +{improvement:.1f} bits")
                    
                    # Switch back to training mode
                    model.train()
                
                # Save checkpoint
                if step % config['evaluation']['save_checkpoint_interval'] == 0:
                    save_model_checkpoint(model, optimizer, step, experiment_id, log_dirs['checkpoints'])
                    print(f"💾 Saved checkpoint at step {step}")
            
            # End of epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"📅 Completed epoch {current_epoch} in {epoch_time/60:.1f} minutes")
            epoch_start_time = time.time()
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Training interrupted at step {step}")
            break
        except Exception as e:
            print(f"❌ Error at step {step}: {e}")
            # Save emergency checkpoint
            save_model_checkpoint(model, optimizer, step, experiment_id, log_dirs['checkpoints'])
            raise
    
    print(f"🎉 Fixed memorization training completed at step {step}")
    
    # Final evaluation
    print("🔍 Performing final memorization evaluation...")
    model.eval()
    
    with torch.no_grad():
        final_metrics = evaluate_memorization_on_dataset(
            model, eval_dataset, eval_metadata, device
        )
    
    # Save final checkpoint
    save_model_checkpoint(model, optimizer, step, experiment_id, log_dirs['checkpoints'])
    
    # Training results with memorization history
    results = {
        'experiment_id': experiment_id,
        'final_step': step,
        'final_epoch': current_epoch,
        'model_config': config['model'],
        'final_memorization': final_metrics,
        'memorization_history': memorization_history,
        'training_completed': True,
        'experiment_type': 'fixed_memorization'
    }
    
    return results

def train_model_with_memorization_tracking(
    config: Dict[str, Any],
    experiment_id: str,
    log_dirs: Dict[str, Path],
    train_dataset: torch.Tensor,
    train_metadata: Dict[str, Any],
    eval_dataset: torch.Tensor,
    eval_metadata: Dict[str, Any],
    resume_from_step: int = 0
) -> Dict[str, Any]:
    """Train model with comprehensive memorization tracking.
    
    Automatically detects fixed memorization mode and uses appropriate training.
    
    Args:
        config: Training configuration
        experiment_id: Unique experiment identifier
        log_dirs: Logging directory paths
        train_dataset: Training dataset tensor
        train_metadata: Training dataset metadata
        eval_dataset: Evaluation dataset tensor
        eval_metadata: Evaluation dataset metadata
        resume_from_step: Step to resume from (0 for new training)
        
    Returns:
        Training results and final metrics
    """
    # Check if this is fixed memorization mode
    is_fixed_memorization = (
        config.get('data', {}).get('memorization_mode', False) or
        train_metadata.get('memorization_info', {}).get('dataset_type') == 'fixed_memorization'
    )
    
    if is_fixed_memorization:
        print("🎯 Detected fixed memorization mode - using fixed dataset training")
        return train_fixed_memorization_model(
            config, experiment_id, log_dirs, 
            train_dataset, train_metadata,
            eval_dataset, eval_metadata,
            resume_from_step
        )
    else:
        print("🔄 Using standard random data training")
        # Standard training loop (original implementation)
        device = config['training']['device']
        
        # Set random seed for reproducibility
        torch.manual_seed(config['experiment']['seed'])
        if device == 'cuda':
            torch.cuda.manual_seed(config['experiment']['seed'])
        
        # Create model
        model = create_gpt_model(
            n_layers=config['model']['n_layers'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            vocab_size=config['model']['vocab_size'],
            seq_length=config['model']['seq_length'],
            device=device
        )
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        # Resume from checkpoint if needed
        start_step = resume_from_step
        if start_step > 0:
            loaded_step, success = load_model_checkpoint(
                model, optimizer, experiment_id, log_dirs['checkpoints']
            )
            if success:
                start_step = loaded_step
                print(f"Resumed from checkpoint at step {start_step}")
            else:
                print(f"Failed to load checkpoint, starting from step 0")
                start_step = 0
        
        # Create data loader
        train_loader = create_dataloader(
            train_dataset, 
            config['training']['batch_size'], 
            shuffle=True, 
            device=device
        )
        
        # Training state
        model.train()
        running_loss = 0.0
        step = start_step
        
        # Training loop
        print(f"Starting training from step {start_step} to {config['training']['max_steps']}")
        
        # Create infinite iterator for training data
        train_iter = iter(train_loader)
        
        while step < config['training']['max_steps']:
            try:
                # Get next batch
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # Restart iterator when epoch ends
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                
                batch = batch.to(device)
                
                # Compute learning rate
                lr = compute_learning_rate(
                    step, 
                    config['training']['learning_rate'],
                    config['training']['warmup_steps'],
                    config['training']['max_steps']
                )
                
                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Forward pass
                logits = model(batch)
                
                # Compute loss (autoregressive language modeling)
                input_ids = batch[:, :-1]  # All but last token
                target_ids = batch[:, 1:]  # All but first token
                logits = logits[:, :-1, :]  # Match target length
                
                # Flatten for cross-entropy
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if config['training']['grad_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config['training']['grad_clip_norm']
                    )
                
                optimizer.step()
                
                # Update running loss
                running_loss += loss.item()
                step += 1
                
                # Logging and evaluation
                if step % config['evaluation']['eval_interval'] == 0:
                    avg_loss = running_loss / config['evaluation']['eval_interval']
                    
                    # Log training step
                    log_training_step(experiment_id, step, avg_loss, 0.0, log_dirs['training'])
                    
                    print(f"Step {step}/{config['training']['max_steps']}: "
                          f"Loss={avg_loss:.4f}, LR={lr:.6f}")
                    
                    running_loss = 0.0
                
                # Memorization evaluation
                if step % config['evaluation']['memorization_eval_interval'] == 0:
                    print(f"Evaluating memorization at step {step}...")
                    
                    # Switch to eval mode
                    model.eval()
                    
                    with torch.no_grad():
                        # Track memorization on evaluation dataset
                        memo_metrics = track_memorization_during_training(
                            model, eval_dataset, eval_metadata, step,
                            experiment_id, log_dirs['training'], device
                        )
                        
                        print(f"  Morris memorization: {memo_metrics['morris_memorization_bits']:.2f} bits")
                        print(f"  Bits per parameter: {memo_metrics['bits_per_parameter']:.4f}")
                        print(f"  Memorization fraction: {memo_metrics['memorization_fraction']:.4f}")
                    
                    # Switch back to training mode
                    model.train()
                
                # Save checkpoint
                if step % config['evaluation']['save_checkpoint_interval'] == 0:
                    save_model_checkpoint(model, optimizer, step, experiment_id, log_dirs['checkpoints'])
                    print(f"Saved checkpoint at step {step}")
                
            except KeyboardInterrupt:
                print(f"\nTraining interrupted at step {step}")
                break
            except Exception as e:
                print(f"Error at step {step}: {e}")
                # Save emergency checkpoint
                save_model_checkpoint(model, optimizer, step, experiment_id, log_dirs['checkpoints'])
                raise
        
        print(f"Training completed at step {step}")
        
        # Final evaluation
        print("Performing final memorization evaluation...")
        model.eval()
        
        with torch.no_grad():
            final_metrics = evaluate_memorization_on_dataset(
                model, eval_dataset, eval_metadata, device
            )
        
        # Save final checkpoint
        save_model_checkpoint(model, optimizer, step, experiment_id, log_dirs['checkpoints'])
        
        # Training results
        results = {
            'experiment_id': experiment_id,
            'final_step': step,
            'model_config': config['model'],
            'final_memorization': final_metrics,
            'training_completed': True
        }
        
        return results

# Keep all the existing functions from the original training_loop.py
# (run_morris_validation_experiment, analyze_morris_scaling_law, etc.)
# They remain unchanged

def run_morris_validation_experiment(
    experiment_name: str = "morris_validation",
    model_names: List[str] = ['nano', 'micro', 'mini'],
    dataset_sizes: List[int] = [500, 1000, 2000],
    base_config_overrides: Optional[Dict[str, Any]] = None,
    log_base_dir: Path = Path("logs"),
    cache_dir: Optional[Path] = Path("data_cache"),
    resume_incomplete: bool = True
) -> Dict[str, Any]:
    """Run complete Morris validation experiment across model sizes.
    
    Args:
        experiment_name: Name for the experiment suite
        model_names: List of model sizes to test
        dataset_sizes: List of dataset sizes to test
        base_config_overrides: Override default config parameters
        log_base_dir: Base directory for logs
        cache_dir: Directory for dataset caching
        resume_incomplete: Whether to resume incomplete experiments
        
    Returns:
        Complete experimental results and analysis
    """
    print(f"🚀 Starting Morris Validation Experiment: {experiment_name}")
    print("=" * 60)
    
    # Setup logging directories
    log_dirs = setup_logging_directories(log_base_dir)
    
    # Main experiment ID
    main_experiment_id = generate_experiment_id()
    
    # Check for incomplete experiments
    if resume_incomplete:
        incomplete_experiments = find_incomplete_experiments(log_dirs['checkpoints'])
        if incomplete_experiments:
            print(f"Found {len(incomplete_experiments)} incomplete experiments")
            for exp_id in incomplete_experiments:
                progress = get_experiment_progress(exp_id, log_dirs['checkpoints'])
                print(f"  - {exp_id}: {progress['current_phase']}, "
                      f"step {progress['latest_checkpoint_step']}")
    
    # Experiment configuration
    experiment_config = {
        'experiment_name': experiment_name,
        'main_experiment_id': main_experiment_id,
        'model_names': model_names,
        'dataset_sizes': dataset_sizes,
        'total_experiments': len(model_names) * len(dataset_sizes),
        'base_config': base_config_overrides or {}
    }
    
    # Log main experiment metadata
    log_experiment_metadata(main_experiment_id, experiment_config, log_dirs['metadata'])
    
    # Results storage
    all_results = []
    experiment_count = 0
    
    # Run experiments for each model size and dataset size
    for model_name in model_names:
        print(f"\n📊 Testing Model: {model_name.upper()}")
        print("-" * 40)
        
        for dataset_size in dataset_sizes:
            experiment_count += 1
            print(f"\n🔬 Experiment {experiment_count}/{experiment_config['total_experiments']}")
            print(f"Model: {model_name}, Dataset Size: {dataset_size}")
            
            # Create experiment configuration
            config = create_training_config(
                model_name=model_name,
                dataset_size=dataset_size,
                **(base_config_overrides or {})
            )
            
            # Generate unique experiment ID for this run
            exp_id = generate_experiment_id()
            
            try:
                # Generate datasets
                print("📥 Generating datasets...")
                train_dataset, train_metadata = get_or_generate_dataset(
                    n_sequences=dataset_size,
                    seq_length=config['data']['seq_length'],
                    vocab_size=config['data']['vocab_size'],
                    seed=config['data']['seed'],
                    cache_dir=cache_dir,
                    use_cache=True
                )
                
                eval_dataset, eval_metadata = get_or_generate_dataset(
                    n_sequences=config['data']['eval_dataset_size'],
                    seq_length=config['data']['seq_length'],
                    vocab_size=config['data']['vocab_size'],
                    seed=config['data']['seed'] + 1000,  # Different seed
                    cache_dir=cache_dir,
                    use_cache=True
                )
                
                # Log experiment configuration
                exp_config = {
                    **config,
                    'parent_experiment_id': main_experiment_id,
                    'experiment_number': experiment_count
                }
                log_experiment_metadata(exp_id, exp_config, log_dirs['metadata'])
                
                # Save experiment state
                experiment_state = {
                    'current_phase': 'training',
                    'model_name': model_name,
                    'dataset_size': dataset_size,
                    'total_experiments': experiment_config['total_experiments'],
                    'experiment_number': experiment_count,
                    'completed': False
                }
                save_experiment_state(exp_id, experiment_state, log_dirs['checkpoints'])
                
                # Train model
                print("🎯 Training model...")
                training_results = train_model_with_memorization_tracking(
                    config=config,
                    experiment_id=exp_id,
                    log_dirs=log_dirs,
                    train_dataset=train_dataset,
                    train_metadata=train_metadata,
                    eval_dataset=eval_dataset,
                    eval_metadata=eval_metadata
                )
                
                # Update experiment state
                experiment_state.update({
                    'current_phase': 'completed',
                    'completed': True,
                    'final_results': training_results
                })
                save_experiment_state(exp_id, experiment_state, log_dirs['checkpoints'])
                mark_experiment_complete(exp_id, log_dirs['checkpoints'])
                
                # Store results
                all_results.append({
                    'experiment_id': exp_id,
                    'model_name': model_name,
                    'dataset_size': dataset_size,
                    **training_results
                })
                
                # Print summary
                final_memo = training_results['final_memorization']
                print(f"✅ Completed: {final_memo['morris_memorization_bits']:.1f} bits, "
                      f"{final_memo['bits_per_parameter']:.3f} bits/param")
                
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                # Update state to reflect failure
                experiment_state.update({
                    'current_phase': 'failed',
                    'error': str(e),
                    'completed': False
                })
                save_experiment_state(exp_id, experiment_state, log_dirs['checkpoints'])
                continue
    
    # Generate comprehensive analysis
    print(f"\n📈 Generating Morris Validation Analysis")
    print("=" * 50)
    
    if all_results:
        # Create memorization report
        evaluation_results = [r['final_memorization'] for r in all_results]
        model_configs = {r['model_name']: r['model_config'] for r in all_results}
        
        analysis_report = create_memorization_report(evaluation_results, model_configs)
        
        # Add Morris-specific analysis
        morris_analysis = analyze_morris_scaling_law(all_results)
        analysis_report['morris_validation'] = morris_analysis
        
        # Safe access to validation results
        passes_validation = False
        if 'scaling_law_validation' in morris_analysis:
            passes_validation = morris_analysis['scaling_law_validation'].get('passes_validation', False)
        
        # Save comprehensive results
        final_results = {
            'experiment_config': experiment_config,
            'individual_results': all_results,
            'analysis_report': analysis_report,
            'summary': {
                'total_experiments': len(all_results),
                'successful_experiments': len([r for r in all_results if r.get('training_completed', False)]),
                'morris_scaling_validated': passes_validation
            }
        }
        
        # Make results JSON serializable
        def make_json_serializable(obj):
            """Convert numpy types and other non-serializable objects to JSON-compatible types."""
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (bool, int, float, str, type(None))):
                return obj
            else:
                return str(obj)  # Convert to string as fallback
        
        # Make results JSON serializable
        json_safe_results = make_json_serializable(final_results)
        
        log_experiment_metadata(
            f"{main_experiment_id}_final_results", 
            json_safe_results, 
            log_dirs['metadata']
        )
        
        print(f"🎉 Morris Validation Complete!")
        print(f"   Experiments completed: {len(all_results)}")
        print(f"   Average bits/parameter: {analysis_report['summary']['average_bits_per_param']:.3f}")
        print(f"   Morris 3.6 validation: {'✅ PASSED' if morris_analysis['scaling_law_validation']['passes_validation'] else '❌ FAILED'}")
        
        return final_results
        
    else:
        print("❌ No experiments completed successfully")
        return {'error': 'No experiments completed', 'results': []}

def analyze_morris_scaling_law(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze results to validate Morris 3.6 bits-per-parameter scaling law.
    
    Args:
        results: List of experimental results
        
    Returns:
        Morris scaling law analysis
    """
    if not results:
        return {'error': 'No results to analyze'}
    
    # Extract data for analysis
    model_params = []
    memorization_bits = []
    bits_per_param = []
    model_names = []
    
    for result in results:
        if 'final_memorization' in result and 'model_config' in result:
            memo = result['final_memorization']
            config = result['model_config']
            
            # Skip results with zero or negative memorization for log analysis
            memo_bits = memo['morris_memorization_bits']
            if memo_bits > 0:
                model_params.append(config['total_params'])
                memorization_bits.append(memo_bits)
                bits_per_param.append(memo['bits_per_parameter'])
                model_names.append(result['model_name'])
    
    if len(model_params) < 2:
        return {
            'error': f'Need at least 2 data points with positive memorization for scaling analysis. Got {len(model_params)} valid points.',
            'total_results': len(results),
            'valid_results': len(model_params)
        }
    
    # Statistical analysis
    import numpy as np
    
    # Linear regression in log space: log(memorization) = α * log(params) + β
    log_params = np.log(model_params)
    log_memo = np.log(memorization_bits)
    
    # Fit power law
    A = np.vstack([log_params, np.ones(len(log_params))]).T
    scaling_exponent, log_intercept = np.linalg.lstsq(A, log_memo, rcond=None)[0]
    
    # R-squared calculation
    log_memo_pred = scaling_exponent * log_params + log_intercept
    ss_res = np.sum((log_memo - log_memo_pred) ** 2)
    ss_tot = np.sum((log_memo - np.mean(log_memo)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Morris 3.6 bits/parameter validation
    avg_bits_per_param = float(np.mean(bits_per_param))
    std_bits_per_param = float(np.std(bits_per_param))
    
    # Efficiency scores relative to 3.6 bits/param
    efficiency_scores = [
        memorization_efficiency_score(memo, params, 3.6) 
        for memo, params in zip(memorization_bits, model_params)
    ]
    
    # Validation criteria
    morris_target = 3.6
    tolerance = 0.5  # ±0.5 bits/parameter tolerance
    
    scaling_validation = {
        'passes_validation': bool(
            abs(avg_bits_per_param - morris_target) < tolerance and
            scaling_exponent > 0.8 and  # Should scale roughly linearly
            r_squared > 0.8  # Good fit
        ),
        'average_bits_per_param': float(avg_bits_per_param),
        'std_bits_per_param': float(std_bits_per_param),
        'target_bits_per_param': float(morris_target),
        'tolerance': float(tolerance),
        'scaling_exponent': float(scaling_exponent),
        'r_squared': float(r_squared)
    }
    
    analysis = {
        'scaling_law_validation': scaling_validation,
        'data_points': int(len(results)),
        'valid_data_points': int(len(model_params)),
        'model_parameters_range': [int(min(model_params)), int(max(model_params))],
        'memorization_range': [float(min(memorization_bits)), float(max(memorization_bits))],
        'efficiency_scores': {
            'mean': float(np.mean(efficiency_scores)),
            'std': float(np.std(efficiency_scores)),
            'individual': [(name, float(score)) for name, score in zip(model_names, efficiency_scores)]
        },
        'power_law_fit': {
            'exponent': float(scaling_exponent),
            'intercept': float(log_intercept),
            'r_squared': float(r_squared),
            'expected_exponent': 1.0  # Linear scaling
        }
    }
    
    return analysis

# Add remaining functions (resume_training_from_checkpoint, etc.) unchanged from original
def quick_morris_test(model_name: str = 'nano', dataset_size: int = 100, 
                     max_steps: int = 200) -> Dict[str, Any]:
    """Run a quick Morris validation test for development/debugging.
    
    Args:
        model_name: Model size to test
        dataset_size: Size of dataset
        max_steps: Number of training steps
        
    Returns:
        Quick test results
    """
    print(f"🧪 Quick Morris Test: {model_name} model, {dataset_size} sequences, {max_steps} steps")
    
    # Create minimal configuration
    config = create_training_config(
        model_name=model_name,
        dataset_size=dataset_size,
        max_steps=max_steps,
        eval_interval=50,
        memorization_eval_interval=100,
        save_checkpoint_interval=max_steps + 1  # No checkpointing
    )
    
    # Setup minimal logging
    log_dirs = setup_logging_directories(Path("logs_quick_test"))
    exp_id = generate_experiment_id()
    
    # Generate small datasets
    train_dataset, train_metadata = get_or_generate_dataset(
        n_sequences=dataset_size,
        seq_length=config['data']['seq_length'],
        vocab_size=config['data']['vocab_size'],
        seed=42,
        cache_dir=None,
        use_cache=False
    )
    
    eval_dataset, eval_metadata = get_or_generate_dataset(
        n_sequences=50,
        seq_length=config['data']['seq_length'],
        vocab_size=config['data']['vocab_size'],
        seed=43,
        cache_dir=None,
        use_cache=False
    )
    
    # Run training
    results = train_model_with_memorization_tracking(
        config=config,
        experiment_id=exp_id,
        log_dirs=log_dirs,
        train_dataset=train_dataset,
        train_metadata=train_metadata,
        eval_dataset=eval_dataset,
        eval_metadata=eval_metadata
    )
    
    print(f"✅ Quick test completed!")
    final_memo = results['final_memorization']
    print(f"   Morris memorization: {final_memo['morris_memorization_bits']:.2f} bits")
    print(f"   Bits per parameter: {final_memo['bits_per_parameter']:.4f}")
    print(f"   Model parameters: {final_memo['model_parameters']:,}")
    
    return results

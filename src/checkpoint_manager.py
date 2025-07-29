# src/checkpoint_manager.py
# Checkpoint management for restartable Morris validation experiments

import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import glob

from logging_utils import atomic_write_json, read_experiment_metadata

def save_experiment_state(experiment_id: str, state: Dict[str, Any], 
                         checkpoint_dir: Path) -> None:
    """Save complete experiment state for resumability.
    
    Args:
        experiment_id: Unique experiment identifier
        state: Complete experiment state dictionary
        checkpoint_dir: Directory to save checkpoint files
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Enrich state with checkpoint metadata
    enriched_state = {
        'experiment_id': experiment_id,
        'checkpoint_timestamp': datetime.now().isoformat(),
        'checkpoint_time': time.time(),
        'state_version': '1.0',
        **state
    }
    
    state_file = checkpoint_dir / f"{experiment_id}_state.json"
    atomic_write_json(enriched_state, state_file)

def load_experiment_state(experiment_id: str, 
                         checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """Load experiment state from checkpoint.
    
    Args:
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Experiment state dictionary if found, None otherwise
    """
    checkpoint_dir = Path(checkpoint_dir)
    state_file = checkpoint_dir / f"{experiment_id}_state.json"
    
    if not state_file.exists():
        return None
    
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         step: int, experiment_id: str, 
                         checkpoint_dir: Path) -> None:
    """Save model and optimizer state during training.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        step: Current training step
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory to save model checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        'experiment_id': experiment_id,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat(),
        'checkpoint_time': time.time()
    }
    
    # Save with step number in filename for multiple checkpoints
    checkpoint_file = checkpoint_dir / f"{experiment_id}_model_step_{step}.pt"
    torch.save(checkpoint_data, checkpoint_file)
    
    # Also save as latest checkpoint
    latest_file = checkpoint_dir / f"{experiment_id}_model_latest.pt"
    torch.save(checkpoint_data, latest_file)

def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         experiment_id: str, checkpoint_dir: Path,
                         step: Optional[int] = None) -> Tuple[int, bool]:
    """Load model and optimizer state from checkpoint.
    
    Args:
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory containing model checkpoints
        step: Specific step to load (latest if None)
        
    Returns:
        Tuple of (loaded_step, success_flag)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Determine checkpoint file to load
    if step is not None:
        checkpoint_file = checkpoint_dir / f"{experiment_id}_model_step_{step}.pt"
    else:
        checkpoint_file = checkpoint_dir / f"{experiment_id}_model_latest.pt"
    
    if not checkpoint_file.exists():
        return 0, False
    
    try:
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
        
        # Verify experiment ID matches
        if checkpoint_data.get('experiment_id') != experiment_id:
            return 0, False
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        loaded_step = checkpoint_data['step']
        return loaded_step, True
        
    except (torch.serialization.pickle.UnpicklingError, KeyError, RuntimeError):
        return 0, False

def find_incomplete_experiments(checkpoint_dir: Path) -> List[str]:
    """Find experiments that can be resumed.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        List of experiment IDs that can be resumed
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    # Find all state files
    state_files = list(checkpoint_dir.glob("*_state.json"))
    
    incomplete_experiments = []
    
    for state_file in state_files:
        # Extract experiment ID from filename
        filename = state_file.stem
        if filename.endswith('_state'):
            experiment_id = filename[:-6]  # Remove '_state' suffix
            
            # Load state to check if experiment is complete
            state = load_experiment_state(experiment_id, checkpoint_dir)
            if state and not state.get('completed', False):
                incomplete_experiments.append(experiment_id)
    
    return incomplete_experiments

def get_experiment_progress(experiment_id: str, 
                          checkpoint_dir: Path) -> Dict[str, Any]:
    """Get current progress of an experiment.
    
    Args:
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Dictionary containing progress information
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load experiment state
    state = load_experiment_state(experiment_id, checkpoint_dir)
    if not state:
        return {'exists': False}
    
    # Find available model checkpoints
    model_checkpoints = list(checkpoint_dir.glob(f"{experiment_id}_model_step_*.pt"))
    checkpoint_steps = []
    
    for checkpoint_file in model_checkpoints:
        # Extract step number from filename
        filename = checkpoint_file.stem
        try:
            step_part = filename.split('_step_')[-1]
            step = int(step_part)
            checkpoint_steps.append(step)
        except (ValueError, IndexError):
            continue
    
    checkpoint_steps.sort()
    
    progress_info = {
        'exists': True,
        'experiment_id': experiment_id,
        'completed': state.get('completed', False),
        'current_phase': state.get('current_phase', 'unknown'),
        'model_checkpoints': checkpoint_steps,
        'latest_checkpoint_step': max(checkpoint_steps) if checkpoint_steps else 0,
        'total_experiments_planned': state.get('total_experiments', 0),
        'experiments_completed': state.get('experiments_completed', 0),
        'last_updated': state.get('checkpoint_timestamp', 'unknown')
    }
    
    return progress_info

def mark_experiment_complete(experiment_id: str, checkpoint_dir: Path) -> None:
    """Mark an experiment as completed.
    
    Args:
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory containing checkpoint files
    """
    state = load_experiment_state(experiment_id, checkpoint_dir)
    if state:
        state['completed'] = True
        state['completion_timestamp'] = datetime.now().isoformat()
        state['completion_time'] = time.time()
        save_experiment_state(experiment_id, state, checkpoint_dir)

def cleanup_old_checkpoints(experiment_id: str, checkpoint_dir: Path, 
                           keep_latest: int = 5) -> None:
    """Clean up old model checkpoints, keeping only the most recent ones.
    
    Args:
        experiment_id: Unique experiment identifier
        checkpoint_dir: Directory containing checkpoint files
        keep_latest: Number of most recent checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find all model checkpoints for this experiment
    model_checkpoints = list(checkpoint_dir.glob(f"{experiment_id}_model_step_*.pt"))
    
    if len(model_checkpoints) <= keep_latest:
        return  # Nothing to clean up
    
    # Sort by step number
    checkpoint_steps = []
    for checkpoint_file in model_checkpoints:
        filename = checkpoint_file.stem
        try:
            step_part = filename.split('_step_')[-1]
            step = int(step_part)
            checkpoint_steps.append((step, checkpoint_file))
        except (ValueError, IndexError):
            continue
    
    checkpoint_steps.sort(key=lambda x: x[0])  # Sort by step number
    
    # Remove older checkpoints
    to_remove = checkpoint_steps[:-keep_latest]
    for _, checkpoint_file in to_remove:
        try:
            checkpoint_file.unlink()
        except OSError:
            pass  # Ignore errors if file already deleted

def get_resumable_experiments(checkpoint_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Get detailed information about all resumable experiments.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Dictionary mapping experiment IDs to their progress information
    """
    incomplete_ids = find_incomplete_experiments(checkpoint_dir)
    
    resumable_info = {}
    for experiment_id in incomplete_ids:
        progress = get_experiment_progress(experiment_id, checkpoint_dir)
        resumable_info[experiment_id] = progress
    
    return resumable_info

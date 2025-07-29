# src/logging_utils.py
# Comprehensive logging utilities for Morris validation experiment

import json
import csv
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil
import threading

# Thread lock for atomic operations
_write_lock = threading.Lock()

def setup_logging_directories(base_path: Path) -> Dict[str, Path]:
    """Create logging directory structure and return paths.
    
    Args:
        base_path: Root directory for logs
        
    Returns:
        Dictionary mapping log types to their directory paths
    """
    base_path = Path(base_path)
    
    # Define subdirectories
    directories = {
        'experiments': base_path / 'experiments',
        'training': base_path / 'training', 
        'checkpoints': base_path / 'checkpoints',
        'metrics': base_path / 'metrics',
        'metadata': base_path / 'metadata'
    }
    
    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories

def generate_experiment_id() -> str:
    """Generate unique experiment ID using timestamp and random component.
    
    Returns:
        Unique experiment identifier string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = str(uuid.uuid4())[:8]
    return f"morris_exp_{timestamp}_{random_suffix}"

def log_experiment_metadata(experiment_id: str, metadata: Dict[str, Any], 
                          log_dir: Path) -> None:
    """Log experiment configuration and metadata to JSON file.
    
    Args:
        experiment_id: Unique experiment identifier
        metadata: Dictionary containing experiment configuration
        log_dir: Directory to save metadata file
    """
    # Add timestamp and experiment ID to metadata
    enriched_metadata = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'created_at': time.time(),
        **metadata
    }
    
    metadata_file = log_dir / f"{experiment_id}_metadata.json"
    atomic_write_json(enriched_metadata, metadata_file)

def log_metrics_csv(experiment_id: str, metrics: Dict[str, Any], 
                   log_dir: Path, append: bool = True) -> None:
    """Log metrics to CSV file with atomic writes.
    
    Args:
        experiment_id: Unique experiment identifier
        metrics: Dictionary of metric values to log
        log_dir: Directory to save CSV file
        append: Whether to append to existing file or create new
    """
    csv_file = log_dir / f"{experiment_id}_metrics.csv"
    
    # Add timestamp to metrics
    timestamped_metrics = {
        'timestamp': datetime.now().isoformat(),
        'unix_time': time.time(),
        **metrics
    }
    
    with _write_lock:
        # Check if file exists and if we need headers
        file_exists = csv_file.exists()
        write_headers = not file_exists or not append
        
        # Use temporary file for atomic write
        with tempfile.NamedTemporaryFile(mode='w', newline='', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # If appending and file exists, copy existing content first
            if append and file_exists:
                with open(csv_file, 'r') as existing_file:
                    temp_file.write(existing_file.read())
            
            # Write new data
            fieldnames = list(timestamped_metrics.keys())
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            
            if write_headers:
                writer.writeheader()
            
            writer.writerow(timestamped_metrics)
        
        # Atomic move
        shutil.move(str(temp_path), str(csv_file))

def log_training_step(experiment_id: str, step: int, loss: float, 
                     morris_bits: float, log_dir: Path) -> None:
    """Log individual training step metrics.
    
    Args:
        experiment_id: Unique experiment identifier
        step: Training step number
        loss: Training loss value
        morris_bits: Morris memorization in bits
        log_dir: Directory to save training logs
    """
    step_metrics = {
        'experiment_id': experiment_id,
        'step': step,
        'loss': loss,
        'morris_bits': morris_bits
    }
    
    log_metrics_csv(experiment_id, step_metrics, log_dir, append=True)

def atomic_write_json(data: Dict[str, Any], file_path: Path) -> None:
    """Write JSON data atomically to prevent corruption.
    
    Args:
        data: Dictionary to write as JSON
        file_path: Target file path
    """
    file_path = Path(file_path)
    
    with _write_lock:
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                       dir=file_path.parent) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
        
        # Atomic move to final location
        shutil.move(str(temp_path), str(file_path))

def read_experiment_metadata(experiment_id: str, log_dir: Path) -> Optional[Dict[str, Any]]:
    """Read experiment metadata from JSON file.
    
    Args:
        experiment_id: Unique experiment identifier
        log_dir: Directory containing metadata files
        
    Returns:
        Metadata dictionary if found, None otherwise
    """
    metadata_file = log_dir / f"{experiment_id}_metadata.json"
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

def read_metrics_csv(experiment_id: str, log_dir: Path) -> List[Dict[str, Any]]:
    """Read metrics from CSV file.
    
    Args:
        experiment_id: Unique experiment identifier  
        log_dir: Directory containing CSV files
        
    Returns:
        List of metric dictionaries
    """
    csv_file = log_dir / f"{experiment_id}_metrics.csv"
    
    if not csv_file.exists():
        return []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except (csv.Error, IOError):
        return []

def get_latest_training_step(experiment_id: str, log_dir: Path) -> Optional[int]:
    """Get the latest training step logged for an experiment.
    
    Args:
        experiment_id: Unique experiment identifier
        log_dir: Directory containing training logs
        
    Returns:
        Latest step number if found, None otherwise
    """
    metrics = read_metrics_csv(experiment_id, log_dir)
    
    if not metrics:
        return None
    
    # Find maximum step number
    max_step = 0
    for metric in metrics:
        if 'step' in metric:
            try:
                step = int(metric['step'])
                max_step = max(max_step, step)
            except (ValueError, TypeError):
                continue
    
    return max_step if max_step > 0 else None

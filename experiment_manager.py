"""
Experiment Manager for Memorization Studies

Provides logging, checkpointing, and restart capabilities for training experiments.
Designed to be testable as a standalone module.
"""

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch


@dataclass
class ExperimentConfig:
    """Configuration for a memorization experiment."""
    experiment_name: str
    model_params: Dict[str, Any]
    data_params: Dict[str, Any]
    training_params: Dict[str, Any]
    output_dir: str
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        return cls(**data)


@dataclass
class ExperimentState:
    """Tracks the current state of an experiment."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    metrics: Dict[str, List[float]] = None
    start_time: float = None
    total_time: float = 0.0
    completed: bool = False
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.start_time is None:
            self.start_time = time.time()


class ExperimentManager:
    """
    Manages experiment logging, checkpointing, and restart functionality.
    
    Features:
    - Structured logging to files with different levels
    - Model and optimizer checkpoint saving/loading
    - Experiment state persistence
    - Progress tracking and metrics logging
    - Automatic directory creation and organization
    """
    
    def __init__(self, config: ExperimentConfig, resume: bool = False):
        self.config = config
        self.experiment_dir = Path(config.output_dir) / config.experiment_name
        
        # Create directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.state = ExperimentState()
        
        # Setup logging
        self._setup_logging()
        
        # Load existing state if resuming
        if resume and self._checkpoint_exists():
            self.load_checkpoint()
            self.logger.info(f"Resumed experiment from step {self.state.step}")
        else:
            self.save_config()
            self.logger.info(f"Started new experiment: {config.experiment_name}")
    
    def _setup_logging(self):
        """Setup file and console logging."""
        # Create logger
        self.logger = logging.getLogger(f"experiment_{self.config.experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers if logger already exists
        if self.logger.handlers:
            return
            
        # File handler for detailed logs
        log_file = self.logs_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important updates
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def save_config(self):
        """Save experiment configuration to file."""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if step is None:
            step = self.state.step
            
        if name not in self.state.metrics:
            self.state.metrics[name] = []
        
        self.state.metrics[name].append(value)
        self.logger.debug(f"Step {step}: {name} = {value:.6f}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                       additional_data: Optional[Dict[str, Any]] = None):
        """Save model, optimizer, and experiment state."""
        checkpoint_path = self.checkpoints_dir / f"checkpoint_step_{self.state.step}.pt"
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'experiment_state': asdict(self.state),
            'config': self.config.to_dict(),
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Also save the latest checkpoint with a consistent name
        latest_path = self.checkpoints_dir / "latest.pt"
        torch.save(checkpoint_data, latest_path)
        
        self.logger.debug(f"Saved checkpoint at step {self.state.step}")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load experiment state and return model/optimizer state dicts."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoints_dir / "latest.pt"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Restore experiment state
        state_dict = checkpoint_data['experiment_state']
        self.state = ExperimentState(**state_dict)
        
        self.logger.info(f"Loaded checkpoint from step {self.state.step}")
        return checkpoint_data
    
    def _checkpoint_exists(self) -> bool:
        """Check if a checkpoint exists for this experiment."""
        return (self.checkpoints_dir / "latest.pt").exists()
    
    def update_step(self, step: Optional[int] = None):
        """Update the current step counter."""
        if step is not None:
            self.state.step = step
        else:
            self.state.step += 1
    
    def start_epoch(self, epoch: int):
        """Mark the start of a new epoch."""
        self.state.epoch = epoch
        self.logger.info(f"Starting epoch {epoch}")
    
    def complete_experiment(self):
        """Mark the experiment as completed."""
        self.state.completed = True
        self.state.total_time = time.time() - self.state.start_time
        
        # Save final metrics summary
        summary_path = self.experiment_dir / "metrics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'final_step': self.state.step,
                'total_epochs': self.state.epoch,
                'total_time_hours': self.state.total_time / 3600,
                'metrics': self.state.metrics,
                'best_loss': self.state.best_loss
            }, f, indent=2)
        
        self.logger.info(f"Experiment completed after {self.state.total_time/3600:.2f} hours")
    
    def log_progress(self, current_step: int, total_steps: int, 
                    current_loss: float, frequency: int = 100):
        """Log training progress at specified frequency."""
        if current_step % frequency == 0:
            progress_pct = (current_step / total_steps) * 100
            elapsed = time.time() - self.state.start_time
            
            self.logger.info(
                f"Step {current_step}/{total_steps} ({progress_pct:.1f}%) - "
                f"Loss: {current_loss:.6f} - Elapsed: {elapsed/60:.1f}m"
            )
            
            # Update best loss
            if current_loss < self.state.best_loss:
                self.state.best_loss = current_loss
                self.logger.info(f"New best loss: {current_loss:.6f}")


def test_experiment_manager():
    """Test the ExperimentManager functionality."""
    import tempfile
    import shutil
    
    print("Testing ExperimentManager...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Test 1: Create new experiment
        config = ExperimentConfig(
            experiment_name="test_memorization",
            model_params={"n_layers": 2, "d_model": 128},
            data_params={"vocab_size": 2048, "seq_length": 64},
            training_params={"lr": 1e-3, "batch_size": 32},
            output_dir=temp_dir,
            notes="Test experiment"
        )
        
        manager = ExperimentManager(config)
        print("✓ Created new experiment manager")
        
        # Test 2: Log some metrics
        manager.log_metric("loss", 2.5)
        manager.log_metrics({"accuracy": 0.75, "perplexity": 12.2})
        print("✓ Logged metrics")
        
        # Test 3: Create dummy model and optimizer for checkpoint test
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        manager.update_step(100)
        manager.save_checkpoint(model, optimizer, {"extra_info": "test"})
        print("✓ Saved checkpoint")
        
        # Test 4: Test loading checkpoint
        manager2 = ExperimentManager(config, resume=True)
        checkpoint_data = manager2.load_checkpoint()
        assert manager2.state.step == 100
        print("✓ Loaded checkpoint and resumed state")
        
        # Test 5: Test completion
        manager2.complete_experiment()
        print("✓ Completed experiment")
        
        print("\nAll tests passed! 🎉")
        
        # Show created files
        experiment_dir = Path(temp_dir) / "test_memorization"
        print(f"\nCreated files in {experiment_dir}:")
        for file_path in experiment_dir.rglob("*"):
            if file_path.is_file():
                print(f"  {file_path.relative_to(experiment_dir)}")
                
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    test_experiment_manager()

# tests/test_checkpoint_manager.py
# Comprehensive tests for checkpoint management

import pytest
import tempfile
import shutil
import torch
import torch.nn as nn
from pathlib import Path
import time

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from checkpoint_manager import (
    save_experiment_state,
    load_experiment_state,
    save_model_checkpoint,
    load_model_checkpoint,
    find_incomplete_experiments,
    get_experiment_progress,
    mark_experiment_complete,
    cleanup_old_checkpoints,
    get_resumable_experiments
)

class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

class TestCheckpointManager:
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_and_optimizer(self):
        """Create test model and optimizer."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model, optimizer
    
    def test_save_load_experiment_state(self, temp_dir):
        """Test saving and loading experiment state."""
        experiment_id = "test_exp_state"
        original_state = {
            'current_phase': 'training',
            'model_size': 1000,
            'dataset_size': 500,
            'completed_experiments': 5,
            'total_experiments': 20,
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 32
            }
        }
        
        # Save state
        save_experiment_state(experiment_id, original_state, temp_dir)
        
        # Check file exists
        state_file = temp_dir / f"{experiment_id}_state.json"
        assert state_file.exists()
        
        # Load state
        loaded_state = load_experiment_state(experiment_id, temp_dir)
        
        assert loaded_state is not None
        assert loaded_state['experiment_id'] == experiment_id
        assert 'checkpoint_timestamp' in loaded_state
        assert 'checkpoint_time' in loaded_state
        assert 'state_version' in loaded_state
        
        # Check original state preserved
        for key, value in original_state.items():
            assert loaded_state[key] == value
        
        # Test non-existent state
        nonexistent_state = load_experiment_state("nonexistent", temp_dir)
        assert nonexistent_state is None
    
    def test_save_load_model_checkpoint(self, temp_dir, model_and_optimizer):
        """Test saving and loading model checkpoints."""
        model, optimizer = model_and_optimizer
        experiment_id = "test_model_checkpoint"
        step = 100
        
        # Get initial model parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Save checkpoint
        save_model_checkpoint(model, optimizer, step, experiment_id, temp_dir)
        
        # Check files exist
        step_file = temp_dir / f"{experiment_id}_model_step_{step}.pt"
        latest_file = temp_dir / f"{experiment_id}_model_latest.pt"
        assert step_file.exists()
        assert latest_file.exists()
        
        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(999.0)
        
        # Load checkpoint
        loaded_step, success = load_model_checkpoint(model, optimizer, experiment_id, temp_dir)
        
        assert success
        assert loaded_step == step
        
        # Check parameters restored
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, initial_params[name])
    
    def test_load_specific_step_checkpoint(self, temp_dir, model_and_optimizer):
        """Test loading checkpoint from specific step."""
        model, optimizer = model_and_optimizer
        experiment_id = "test_specific_step"
        
        # Save multiple checkpoints
        steps = [10, 20, 30]
        step_params = {}
        
        for step in steps:
            # Set unique parameter values for each step
            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    param.fill_(step + i)
            
            # Save current parameters
            step_params[step] = {name: param.clone() for name, param in model.named_parameters()}
            
            # Save checkpoint
            save_model_checkpoint(model, optimizer, step, experiment_id, temp_dir)
        
        # Load specific step checkpoint
        target_step = 20
        loaded_step, success = load_model_checkpoint(model, optimizer, experiment_id, temp_dir, step=target_step)
        
        assert success
        assert loaded_step == target_step
        
        # Check correct parameters loaded
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, step_params[target_step][name])
    
    def test_load_nonexistent_checkpoint(self, temp_dir, model_and_optimizer):
        """Test loading non-existent checkpoint."""
        model, optimizer = model_and_optimizer
        experiment_id = "nonexistent_exp"
        
        loaded_step, success = load_model_checkpoint(model, optimizer, experiment_id, temp_dir)
        
        assert not success
        assert loaded_step == 0
    
    def test_find_incomplete_experiments(self, temp_dir):
        """Test finding incomplete experiments."""
        # Create several experiment states
        experiments = [
            ("exp_complete", {"completed": True}),
            ("exp_incomplete_1", {"completed": False, "current_phase": "training"}),
            ("exp_incomplete_2", {"current_phase": "data_prep"}),  # No completed field
            ("exp_complete_2", {"completed": True, "current_phase": "finished"})
        ]
        
        for exp_id, state in experiments:
            save_experiment_state(exp_id, state, temp_dir)
        
        # Find incomplete experiments
        incomplete = find_incomplete_experiments(temp_dir)
        
        expected_incomplete = ["exp_incomplete_1", "exp_incomplete_2"]
        assert set(incomplete) == set(expected_incomplete)
    
    def test_get_experiment_progress(self, temp_dir, model_and_optimizer):
        """Test getting experiment progress."""
        model, optimizer = model_and_optimizer
        experiment_id = "test_progress"
        
        # Save experiment state
        state = {
            'current_phase': 'training',
            'total_experiments': 10,
            'experiments_completed': 3,
            'completed': False
        }
        save_experiment_state(experiment_id, state, temp_dir)
        
        # Save some model checkpoints
        checkpoint_steps = [10, 20, 30, 25, 15]  # Not in order
        for step in checkpoint_steps:
            save_model_checkpoint(model, optimizer, step, experiment_id, temp_dir)
        
        # Get progress
        progress = get_experiment_progress(experiment_id, temp_dir)
        
        assert progress['exists']
        assert progress['experiment_id'] == experiment_id
        assert not progress['completed']
        assert progress['current_phase'] == 'training'
        assert progress['total_experiments_planned'] == 10
        assert progress['experiments_completed'] == 3
        assert progress['latest_checkpoint_step'] == 30  # Maximum step
        assert set(progress['model_checkpoints']) == set(checkpoint_steps)
        
        # Test non-existent experiment
        nonexistent_progress = get_experiment_progress("nonexistent", temp_dir)
        assert not nonexistent_progress['exists']
    
    def test_mark_experiment_complete(self, temp_dir):
        """Test marking experiment as complete."""
        experiment_id = "test_complete"
        
        # Save incomplete experiment
        state = {'current_phase': 'training', 'completed': False}
        save_experiment_state(experiment_id, state, temp_dir)
        
        # Mark as complete
        mark_experiment_complete(experiment_id, temp_dir)
        
        # Check state updated
        updated_state = load_experiment_state(experiment_id, temp_dir)
        assert updated_state['completed']
        assert 'completion_timestamp' in updated_state
        assert 'completion_time' in updated_state
    
    def test_cleanup_old_checkpoints(self, temp_dir, model_and_optimizer):
        """Test cleanup of old checkpoints."""
        model, optimizer = model_and_optimizer
        experiment_id = "test_cleanup"
        
        # Save many checkpoints
        steps = list(range(10, 101, 10))  # 10, 20, 30, ..., 100
        for step in steps:
            save_model_checkpoint(model, optimizer, step, experiment_id, temp_dir)
        
        # Check all checkpoints exist
        checkpoint_files = list(temp_dir.glob(f"{experiment_id}_model_step_*.pt"))
        assert len(checkpoint_files) == len(steps)
        
        # Cleanup, keeping only 3 most recent
        cleanup_old_checkpoints(experiment_id, temp_dir, keep_latest=3)
        
        # Check only 3 remain
        remaining_files = list(temp_dir.glob(f"{experiment_id}_model_step_*.pt"))
        assert len(remaining_files) == 3
        
        # Check correct files remain (highest step numbers)
        remaining_steps = []
        for file in remaining_files:
            step = int(file.stem.split('_step_')[-1])
            remaining_steps.append(step)
        
        assert set(remaining_steps) == {80, 90, 100}
        
        # Check latest file still exists
        latest_file = temp_dir / f"{experiment_id}_model_latest.pt"
        assert latest_file.exists()
    
    def test_get_resumable_experiments(self, temp_dir):
        """Test getting all resumable experiments."""
        # Create mix of complete and incomplete experiments
        experiments = [
            ("exp_complete", {"completed": True}),
            ("exp_incomplete_1", {"completed": False, "current_phase": "training", "total_experiments": 5, "experiments_completed": 2}),
            ("exp_incomplete_2", {"current_phase": "data_prep", "total_experiments": 3, "experiments_completed": 0}),
        ]
        
        for exp_id, state in experiments:
            save_experiment_state(exp_id, state, temp_dir)
        
        # Get resumable experiments
        resumable = get_resumable_experiments(temp_dir)
        
        assert len(resumable) == 2
        assert "exp_complete" not in resumable
        assert "exp_incomplete_1" in resumable
        assert "exp_incomplete_2" in resumable
        
        # Check progress info included
        exp1_progress = resumable["exp_incomplete_1"]
        assert exp1_progress['exists']
        assert not exp1_progress['completed']
        assert exp1_progress['current_phase'] == 'training'
    
    def test_checkpoint_experiment_id_validation(self, temp_dir, model_and_optimizer):
        """Test that checkpoint loading validates experiment ID."""
        model, optimizer = model_and_optimizer
        correct_id = "correct_exp"
        wrong_id = "wrong_exp"
        step = 50
        
        # Save checkpoint with correct ID
        save_model_checkpoint(model, optimizer, step, correct_id, temp_dir)
        
        # Try to load with wrong ID
        loaded_step, success = load_model_checkpoint(model, optimizer, wrong_id, temp_dir)
        assert not success
        assert loaded_step == 0
        
        # Load with correct ID should work
        loaded_step, success = load_model_checkpoint(model, optimizer, correct_id, temp_dir)
        assert success
        assert loaded_step == step
    
    def test_state_persistence_across_interruptions(self, temp_dir):
        """Test that state persists correctly across simulated interruptions."""
        experiment_id = "test_interruption"
        
        # Simulate partial experiment
        initial_state = {
            'current_phase': 'training',
            'total_experiments': 10,
            'experiments_completed': 3,
            'current_model_size': 1000,
            'current_dataset_size': 500
        }
        
        save_experiment_state(experiment_id, initial_state, temp_dir)
        
        # Simulate continuing after interruption
        loaded_state = load_experiment_state(experiment_id, temp_dir)
        assert loaded_state is not None
        
        # Update progress
        loaded_state['experiments_completed'] = 7
        loaded_state['current_phase'] = 'validation'
        
        save_experiment_state(experiment_id, loaded_state, temp_dir)
        
        # Load again to verify persistence
        final_state = load_experiment_state(experiment_id, temp_dir)
        assert final_state['experiments_completed'] == 7
        assert final_state['current_phase'] == 'validation'
        
        # Original fields should still be present
        assert final_state['total_experiments'] == 10
        assert final_state['current_model_size'] == 1000

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_integration_checkpoint_workflow(temp_dir):
    """Test complete checkpoint workflow integration."""
    model = SimpleTestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    experiment_id = "test_integration"
    
    # Start experiment
    initial_state = {
        'current_phase': 'training',
        'total_experiments': 3,
        'experiments_completed': 0,
        'completed': False
    }
    save_experiment_state(experiment_id, initial_state, temp_dir)
    
    # Simulate training with checkpoints
    for step in [10, 20, 30]:
        # Modify model slightly
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)
        
        save_model_checkpoint(model, optimizer, step, experiment_id, temp_dir)
        
        # Update progress
        state = load_experiment_state(experiment_id, temp_dir)
        state['experiments_completed'] = step // 10
        save_experiment_state(experiment_id, state, temp_dir)
    
    # Simulate interruption and resume
    progress = get_experiment_progress(experiment_id, temp_dir)
    assert progress['latest_checkpoint_step'] == 30
    assert progress['experiments_completed'] == 3
    
    # Resume from checkpoint
    new_model = SimpleTestModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    
    loaded_step, success = load_model_checkpoint(new_model, new_optimizer, experiment_id, temp_dir)
    assert success
    assert loaded_step == 30
    
    # Complete experiment
    mark_experiment_complete(experiment_id, temp_dir)
    
    # Verify completion
    final_progress = get_experiment_progress(experiment_id, temp_dir)
    assert final_progress['completed']
    
    # Should not appear in resumable experiments
    resumable = get_resumable_experiments(temp_dir)
    assert experiment_id not in resumable

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

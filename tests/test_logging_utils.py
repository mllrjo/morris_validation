# tests/test_logging_utils.py
# Comprehensive tests for logging utilities

import pytest
import tempfile
import shutil
import json
import csv
from pathlib import Path
from unittest.mock import patch
import time

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from logging_utils import (
    setup_logging_directories,
    generate_experiment_id,
    log_experiment_metadata,
    log_metrics_csv,
    log_training_step,
    atomic_write_json,
    read_experiment_metadata,
    read_metrics_csv,
    get_latest_training_step
)

class TestLoggingUtils:
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_setup_logging_directories(self, temp_dir):
        """Test logging directory setup."""
        directories = setup_logging_directories(temp_dir)
        
        # Check all expected directories are returned
        expected_dirs = ['experiments', 'training', 'checkpoints', 'metrics', 'metadata']
        assert set(directories.keys()) == set(expected_dirs)
        
        # Check all directories exist
        for dir_path in directories.values():
            assert dir_path.exists()
            assert dir_path.is_dir()
        
        # Check directory structure
        for dir_name in expected_dirs:
            expected_path = temp_dir / dir_name
            assert expected_path.exists()
            assert directories[dir_name] == expected_path
    
    def test_generate_experiment_id(self):
        """Test experiment ID generation."""
        # Generate multiple IDs
        id1 = generate_experiment_id()
        time.sleep(0.01)  # Ensure different timestamp
        id2 = generate_experiment_id()
        
        # Check format
        assert id1.startswith('morris_exp_')
        assert id2.startswith('morris_exp_')
        
        # Check uniqueness
        assert id1 != id2
        
        # Check components
        parts1 = id1.split('_')
        assert len(parts1) == 5  # morris, exp, date, time, uuid
        assert parts1[0] == 'morris'
        assert parts1[1] == 'exp'
        
        # Check timestamp format (YYYYMMDD and HHMMSS)
        date_part = parts1[2]
        time_part = parts1[3]
        assert len(date_part) == 8  # YYYYMMDD
        assert len(time_part) == 6  # HHMMSS
        
        # Check UUID part
        uuid_part = parts1[4]
        assert len(uuid_part) == 8  # Truncated UUID
    
    def test_log_experiment_metadata(self, temp_dir):
        """Test experiment metadata logging."""
        experiment_id = "test_exp_001"
        metadata = {
            'model_size': 1000,
            'dataset_size': 100,
            'learning_rate': 0.001,
            'nested_config': {'param1': 'value1', 'param2': 42}
        }
        
        log_experiment_metadata(experiment_id, metadata, temp_dir)
        
        # Check file exists
        metadata_file = temp_dir / f"{experiment_id}_metadata.json"
        assert metadata_file.exists()
        
        # Check file content
        with open(metadata_file, 'r') as f:
            logged_data = json.load(f)
        
        # Check enriched metadata
        assert logged_data['experiment_id'] == experiment_id
        assert 'timestamp' in logged_data
        assert 'created_at' in logged_data
        
        # Check original metadata preserved
        for key, value in metadata.items():
            assert logged_data[key] == value
    
    def test_atomic_write_json(self, temp_dir):
        """Test atomic JSON writing."""
        test_file = temp_dir / "test_atomic.json"
        test_data = {
            'key1': 'value1',
            'key2': 42,
            'key3': [1, 2, 3],
            'key4': {'nested': True}
        }
        
        atomic_write_json(test_data, test_file)
        
        # Check file exists and content is correct
        assert test_file.exists()
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data
    
    def test_log_metrics_csv_new_file(self, temp_dir):
        """Test CSV metrics logging for new file."""
        experiment_id = "test_exp_csv"
        metrics = {
            'step': 100,
            'loss': 0.5,
            'accuracy': 0.85,
            'learning_rate': 0.001
        }
        
        log_metrics_csv(experiment_id, metrics, temp_dir, append=False)
        
        # Check file exists
        csv_file = temp_dir / f"{experiment_id}_metrics.csv"
        assert csv_file.exists()
        
        # Check CSV content
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        row = rows[0]
        
        # Check original metrics
        for key, value in metrics.items():
            assert row[key] == str(value)
        
        # Check added timestamp fields
        assert 'timestamp' in row
        assert 'unix_time' in row
    
    def test_log_metrics_csv_append(self, temp_dir):
        """Test CSV metrics logging with append."""
        experiment_id = "test_exp_append"
        
        # Log first metrics
        metrics1 = {'step': 100, 'loss': 0.5}
        log_metrics_csv(experiment_id, metrics1, temp_dir, append=False)
        
        # Log second metrics (append)
        metrics2 = {'step': 200, 'loss': 0.3}
        log_metrics_csv(experiment_id, metrics2, temp_dir, append=True)
        
        # Check file content
        csv_file = temp_dir / f"{experiment_id}_metrics.csv"
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        assert rows[0]['step'] == '100'
        assert rows[1]['step'] == '200'
    
    def test_log_training_step(self, temp_dir):
        """Test training step logging."""
        experiment_id = "test_training"
        
        log_training_step(experiment_id, 50, 0.75, 1250.5, temp_dir)
        
        # Check file exists and content
        csv_file = temp_dir / f"{experiment_id}_metrics.csv"
        assert csv_file.exists()
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        row = rows[0]
        
        assert row['experiment_id'] == experiment_id
        assert row['step'] == '50'
        assert row['loss'] == '0.75'
        assert row['morris_bits'] == '1250.5'
    
    def test_read_experiment_metadata(self, temp_dir):
        """Test reading experiment metadata."""
        experiment_id = "test_read_meta"
        original_metadata = {'param1': 'value1', 'param2': 42}
        
        # Log metadata first
        log_experiment_metadata(experiment_id, original_metadata, temp_dir)
        
        # Read it back
        read_metadata = read_experiment_metadata(experiment_id, temp_dir)
        
        assert read_metadata is not None
        assert read_metadata['experiment_id'] == experiment_id
        for key, value in original_metadata.items():
            assert read_metadata[key] == value
        
        # Test non-existent file
        nonexistent_metadata = read_experiment_metadata("nonexistent", temp_dir)
        assert nonexistent_metadata is None
    
    def test_read_metrics_csv(self, temp_dir):
        """Test reading metrics from CSV."""
        experiment_id = "test_read_csv"
        
        # Log some metrics
        metrics_list = [
            {'step': 10, 'loss': 0.9},
            {'step': 20, 'loss': 0.7},
            {'step': 30, 'loss': 0.5}
        ]
        
        for metrics in metrics_list:
            log_metrics_csv(experiment_id, metrics, temp_dir, append=True)
        
        # Read back
        read_metrics = read_metrics_csv(experiment_id, temp_dir)
        
        assert len(read_metrics) == 3
        for i, metrics in enumerate(read_metrics):
            assert metrics['step'] == str(metrics_list[i]['step'])
            assert metrics['loss'] == str(metrics_list[i]['loss'])
        
        # Test non-existent file
        nonexistent_metrics = read_metrics_csv("nonexistent", temp_dir)
        assert nonexistent_metrics == []
    
    def test_get_latest_training_step(self, temp_dir):
        """Test getting latest training step."""
        experiment_id = "test_latest_step"
        
        # Log multiple training steps
        steps = [10, 50, 25, 75, 60]  # Not in order
        for step in steps:
            log_training_step(experiment_id, step, 0.5, 1000.0, temp_dir)
        
        # Get latest step
        latest_step = get_latest_training_step(experiment_id, temp_dir)
        assert latest_step == 75  # Maximum step
        
        # Test non-existent experiment
        nonexistent_step = get_latest_training_step("nonexistent", temp_dir)
        assert nonexistent_step is None
    
    def test_concurrent_writes(self, temp_dir):
        """Test thread safety of atomic writes."""
        import threading
        import concurrent.futures
        
        experiment_id = "test_concurrent"
        num_threads = 10
        
        def write_metrics(thread_id):
            metrics = {'thread_id': thread_id, 'value': thread_id * 10}
            log_metrics_csv(experiment_id, metrics, temp_dir, append=True)
        
        # Run concurrent writes
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_metrics, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)
        
        # Check all writes succeeded
        read_metrics = read_metrics_csv(experiment_id, temp_dir)
        assert len(read_metrics) == num_threads
        
        # Check all thread IDs are present
        thread_ids = {int(m['thread_id']) for m in read_metrics}
        assert thread_ids == set(range(num_threads))

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_integration_workflow(temp_dir):
    """Test complete logging workflow."""
    # Setup directories
    directories = setup_logging_directories(temp_dir)
    
    # Generate experiment
    experiment_id = generate_experiment_id()
    
    # Log metadata
    metadata = {
        'model_params': 50000,
        'dataset_size': 1000,
        'config': {'lr': 0.001, 'batch_size': 32}
    }
    log_experiment_metadata(experiment_id, metadata, directories['metadata'])
    
    # Log training steps
    for step in range(0, 100, 10):
        loss = 1.0 - (step / 100)  # Decreasing loss
        morris_bits = step * 10    # Increasing memorization
        log_training_step(experiment_id, step, loss, morris_bits, directories['training'])
    
    # Verify complete workflow
    read_meta = read_experiment_metadata(experiment_id, directories['metadata'])
    assert read_meta is not None
    assert read_meta['model_params'] == 50000
    
    read_metrics = read_metrics_csv(experiment_id, directories['training'])
    assert len(read_metrics) == 10
    
    latest_step = get_latest_training_step(experiment_id, directories['training'])
    assert latest_step == 90

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

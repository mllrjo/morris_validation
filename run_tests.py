# run_tests.py - Updated version
import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests and report results."""
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    test_dir = project_root / "tests"
    
    # Add src to Python path
    env = os.environ.copy()
    pythonpath = str(src_path)
    if 'PYTHONPATH' in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env['PYTHONPATH'] = pythonpath
    
    if not test_dir.exists():
        print("Error: tests directory not found")
        return False
    
    test_files = [
        "test_logging_utils.py",
        "test_checkpoint_manager.py", 
        "test_data_generation.py",
        "test_model_architecture.py",
        "test_memorization_metrics.py",
        "test_training_loop.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"Warning: {test_file} not found")
            continue
        
        print(f"\n{'='*50}")
        print(f"Running {test_file}")
        print('='*50)
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v"],
                env=env,  # Pass the modified environment
                capture_output=False,
                check=True,
                cwd=str(project_root)
            )
        except subprocess.CalledProcessError:
            print(f"FAILED: {test_file}")
            all_passed = False
        except Exception as e:
            print(f"ERROR running {test_file}: {e}")
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print('='*50)
    
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

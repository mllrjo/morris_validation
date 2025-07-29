#!/usr/bin/env python3
# Check project status and file structure

from pathlib import Path
import subprocess
import sys

def check_project_structure():
    """Check if all required files exist in correct locations."""
    print("🔍 Checking project structure...")
    
    # Expected structure
    expected_files = {
        'src': [
            'model_architecture.py',
            'data_generation.py', 
            'logging_utils.py',
            'checkpoint_manager.py',
            'memorization_metrics.py'
        ],
        'tests': [
            'test_model_architecture.py',
            'test_data_generation.py',
            'test_logging_utils.py', 
            'test_checkpoint_manager.py',
            'test_memorization_metrics.py'
        ],
        '.': [
            'function_declarations.py',
            'requirements.txt',
            'run_tests.py',
            'setup_project.py'
        ]
    }
    
    missing_files = []
    
    for directory, files in expected_files.items():
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"❌ Directory missing: {directory}")
            missing_files.extend([f"{directory}/{f}" for f in files])
            continue
            
        for file in files:
            file_path = dir_path / file
            if file_path.exists():
                print(f"✅ {directory}/{file}")
            else:
                print(f"❌ {directory}/{file}")
                missing_files.append(f"{directory}/{file}")
    
    return missing_files

def run_memorization_test():
    """Run just the memorization metrics test to check our fix."""
    print("\n🧪 Testing memorization metrics fix...")
    
    try:
        result = subprocess.run([
            sys.executable, "-c", """
import sys
from pathlib import Path
sys.path.append(str(Path('.') / 'src'))

import torch
import torch.nn as nn
from memorization_metrics import calculate_model_conditional_entropy
from data_generation import generate_random_binary_sequences

class SimplePerfectMemoryModel(nn.Module):
    def __init__(self, vocab_size=2, seq_length=8):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        output = torch.zeros(batch_size, seq_len, self.vocab_size)
        
        for i in range(batch_size):
            for j in range(seq_len):
                if j < seq_len - 1:
                    next_token = x[i, j+1].item()
                    output[i, j, next_token] = 10.0
                    for k in range(self.vocab_size):
                        if k != next_token:
                            output[i, j, k] = -10.0
                else:
                    output[i, j, 0] = 10.0
                    for k in range(1, self.vocab_size):
                        output[i, j, k] = -10.0
        return output

# Test
dataset = generate_random_binary_sequences(4, 8, 2, 42)
model = SimplePerfectMemoryModel()
entropy = calculate_model_conditional_entropy(model, dataset, 'cpu', 2)
print(f"Conditional entropy: {entropy}")
print("PASS" if entropy < 0.01 else "FAIL")
"""
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if "PASS" in output:
                print("✅ Memorization metrics fix successful!")
                print(f"   {output}")
                return True
            else:
                print("❌ Memorization metrics test failed")
                print(f"   {output}")
                return False
        else:
            print("❌ Error running test:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Exception during test: {e}")
        return False

def main():
    print("📊 MORRIS VALIDATION PROJECT STATUS")
    print("=" * 50)
    
    # Check file structure
    missing = check_project_structure()
    
    if missing:
        print(f"\n⚠️  Missing files ({len(missing)}):")
        for file in missing:
            print(f"   - {file}")
    else:
        print("\n✅ All required files present")
    
    # Test memorization fix
    test_passed = run_memorization_test()
    
    # Summary
    print("\n📋 STATUS SUMMARY")
    print("=" * 30)
    print(f"Project structure: {'✅ Complete' if not missing else '⚠️  Incomplete'}")
    print(f"Memorization fix: {'✅ Working' if test_passed else '❌ Failed'}")
    
    if not missing and test_passed:
        print("\n🎉 Project ready for training loop implementation!")
        return True
    else:
        print("\n🔧 Fixes needed before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

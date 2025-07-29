#!/usr/bin/env python3
# Simple test to verify imports work

import sys
from pathlib import Path

# Add src to path (correct path: same level as src/, not parent.parent)
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_basic_imports():
    """Test that all modules can be imported."""
    try:
        # Test each module
        import model_architecture
        import data_generation
        import memorization_metrics
        import training_loop
        import logging_utils
        import checkpoint_manager
        
        print("✅ All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality works."""
    try:
        from training_loop import create_training_config
        from model_architecture import get_morris_model_configs
        
        # Test creating config
        config = create_training_config(model_name='nano', max_steps=10)
        print(f"✅ Config created: {config['model']['name']}")
        
        # Test model configs
        configs = get_morris_model_configs()
        print(f"✅ Model configs loaded: {list(configs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Simple Import Verification Test")
    print("=" * 40)
    
    imports_ok = test_basic_imports()
    if imports_ok:
        functionality_ok = test_basic_functionality()
        if functionality_ok:
            print("\n🎉 Everything working correctly!")
        else:
            print("\n⚠️  Imports work but functionality issues")
    else:
        print("\n❌ Import issues need to be resolved")

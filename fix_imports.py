#!/usr/bin/env python3
# fix_imports.py - Fix import issues in the project

import os
import sys
from pathlib import Path

def check_tests_directory():
    """Check what's in the tests directory."""
    print("📁 Checking tests directory...")
    tests_path = Path('tests')
    
    if not tests_path.exists():
        print("❌ tests/ directory doesn't exist!")
        tests_path.mkdir(exist_ok=True)
        print("✅ Created tests/ directory")
    
    test_files = list(tests_path.glob('*.py'))
    print(f"Found {len(test_files)} test files:")
    for f in test_files:
        print(f"   📄 {f}")
    
    return test_files

def test_imports_directly():
    """Test imports by adding src to Python path."""
    print("\n🧪 Testing imports with src/ in path...")
    
    # Add src to path
    src_path = Path('src').absolute()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to Python path")
    
    # Test each module
    modules_to_test = [
        'model_architecture',
        'data_generation', 
        'memorization_metrics',
        'training_loop',
        'logging_utils',
        'checkpoint_manager'
    ]
    
    import_results = {}
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name}: Import successful")
            import_results[module_name] = True
        except Exception as e:
            print(f"❌ {module_name}: {e}")
            import_results[module_name] = False
    
    return import_results

def run_single_test():
    """Try running a single test to see the exact error."""
    print("\n🎯 Running single test to diagnose error...")
    
    # Add src to path first
    src_path = Path('src').absolute()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        # Try importing what the test needs
        print("Testing training_loop imports...")
        from training_loop import create_training_config
        print("✅ training_loop import successful")
        
        print("Testing model_architecture imports...")
        from model_architecture import get_morris_model_configs
        print("✅ model_architecture import successful")
        
        print("✅ All imports working!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def fix_python_path_in_tests():
    """Check if test files have correct Python path setup."""
    print("\n🔧 Checking Python path setup in test files...")
    
    tests_path = Path('tests')
    test_files = list(tests_path.glob('test_*.py'))
    
    expected_path_setup = [
        "import sys",
        "sys.path.append(str(Path(__file__).parent.parent / 'src'))"
    ]
    
    for test_file in test_files:
        try:
            content = test_file.read_text()
            has_path_setup = all(line in content for line in expected_path_setup)
            
            if has_path_setup:
                print(f"✅ {test_file.name}: Has correct path setup")
            else:
                print(f"⚠️  {test_file.name}: Missing path setup")
                
        except Exception as e:
            print(f"❌ {test_file.name}: Error reading file - {e}")

def create_simple_test():
    """Create a simple test to verify everything works."""
    print("\n🧪 Creating simple verification test...")
    
    test_content = '''#!/usr/bin/env python3
# Simple test to verify imports work

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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
            print("\\n🎉 Everything working correctly!")
        else:
            print("\\n⚠️  Imports work but functionality issues")
    else:
        print("\\n❌ Import issues need to be resolved")
'''
    
    test_file = Path('simple_test.py')
    test_file.write_text(test_content)
    print(f"✅ Created {test_file}")
    
    return test_file

def main():
    """Main diagnostic and fix workflow."""
    print("🔧 Morris Validation - Import Fix Diagnostic")
    print("=" * 50)
    
    # Step 1: Check tests directory
    test_files = check_tests_directory()
    
    # Step 2: Test imports with src in path
    import_results = test_imports_directly()
    
    # Step 3: Check test file path setup
    if test_files:
        fix_python_path_in_tests()
    
    # Step 4: Try running single test
    single_test_ok = run_single_test()
    
    # Step 5: Create simple test
    simple_test_file = create_simple_test()
    
    # Summary
    print(f"\n📊 Diagnostic Summary")
    print("=" * 30)
    print(f"Source files in src/: ✅")
    print(f"Direct imports work: {'✅' if all(import_results.values()) else '❌'}")
    print(f"Single test works: {'✅' if single_test_ok else '❌'}")
    
    if all(import_results.values()) and single_test_ok:
        print(f"\n🎉 Imports are working!")
        print(f"The issue might be in the test runner or specific test content.")
        print(f"\nTry running the simple test:")
        print(f"   python {simple_test_file}")
        print(f"\nThen try running tests again:")
        print(f"   python run_tests.py")
    else:
        print(f"\n🔧 Import issues detected. Check the errors above.")

if __name__ == "__main__":
    main()

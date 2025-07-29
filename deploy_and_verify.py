#!/usr/bin/env python3
# deploy_and_verify.py
# Script to deploy files and verify project structure

from pathlib import Path
import sys

def check_file_structure():
    """Check current file structure and identify missing files."""
    print("🔍 Checking Project File Structure")
    print("=" * 50)
    
    # Required files and their expected locations
    required_files = {
        'src/model_architecture.py': 'GPT transformer implementation',
        'src/data_generation.py': 'Random binary sequence generation', 
        'src/logging_utils.py': 'Comprehensive logging system',
        'src/checkpoint_manager.py': 'Experiment state management',
        'src/memorization_metrics.py': 'Morris memorization measurement',
        'src/training_loop.py': 'Complete training pipeline',
        'tests/test_model_architecture.py': 'Model component tests',
        'tests/test_data_generation.py': 'Data generation tests',
        'tests/test_logging_utils.py': 'Logging system tests', 
        'tests/test_checkpoint_manager.py': 'Checkpointing tests',
        'tests/test_memorization_metrics.py': 'Memorization measurement tests',
        'tests/test_training_loop.py': 'Training pipeline tests',
        'function_declarations.py': 'Central function registry',
        'requirements.txt': 'Python dependencies',
        'run_tests.py': 'Test runner',
        'setup_project.py': 'Project structure setup'
    }
    
    missing_files = []
    present_files = []
    
    for file_path, description in required_files.items():
        path = Path(file_path)
        if path.exists():
            # Check if it has content (not empty)
            try:
                content = path.read_text()
                if len(content.strip()) > 100:  # Reasonable content check
                    present_files.append((file_path, description, "✅"))
                else:
                    present_files.append((file_path, description, "⚠️  (empty)"))
            except:
                present_files.append((file_path, description, "❌ (read error)"))
        else:
            missing_files.append((file_path, description))
    
    # Print results
    print("📁 Present Files:")
    for file_path, description, status in present_files:
        print(f"   {status} {file_path} - {description}")
    
    if missing_files:
        print(f"\n❌ Missing Files ({len(missing_files)}):")
        for file_path, description in missing_files:
            print(f"   📄 {file_path} - {description}")
    else:
        print(f"\n✅ All required files present!")
    
    return missing_files

def identify_file_locations():
    """Identify where source files currently exist."""
    print(f"\n🔍 Locating Existing Files")
    print("=" * 40)
    
    # Look for key files in current directory
    current_files = list(Path('.').glob('*.py'))
    
    key_indicators = {
        'model_architecture': ['class GPTModel', 'MultiHeadAttention', 'create_gpt_model'],
        'data_generation': ['generate_random_binary_sequences', 'calculate_theoretical_entropy'],
        'memorization_metrics': ['calculate_morris_memorization', 'calculate_model_conditional_entropy'],
        'training_loop': ['train_model_with_memorization_tracking', 'run_morris_validation_experiment'],
        'logging_utils': ['setup_logging_directories', 'log_experiment_metadata'],
        'checkpoint_manager': ['save_experiment_state', 'load_experiment_state']
    }
    
    found_sources = {}
    
    for file_path in current_files:
        try:
            content = file_path.read_text()
            for module_name, indicators in key_indicators.items():
                if all(indicator in content for indicator in indicators):
                    found_sources[module_name] = file_path
                    print(f"📍 Found {module_name} source in: {file_path}")
                    break
        except:
            continue
    
    return found_sources

def create_deployment_instructions(missing_files, found_sources):
    """Create specific deployment instructions."""
    print(f"\n📋 Deployment Instructions")
    print("=" * 40)
    
    if not missing_files:
        print("✅ No deployment needed - all files are in place!")
        return
    
    print("Follow these steps to deploy the project correctly:")
    
    # Step 1: Create directories
    print(f"\n1️⃣  Create Directory Structure:")
    print("   python setup_project.py")
    
    # Step 2: File-specific instructions
    print(f"\n2️⃣  Deploy Source Files:")
    
    deployment_map = {
        'src/model_architecture.py': 'model_architecture_fixed artifact OR test_model_architecture.py content',
        'src/training_loop.py': 'training_loop artifact',
        'tests/test_model_architecture.py': 'test_model_architecture artifact (updated)',
        'tests/test_training_loop.py': 'test_training_loop artifact'
    }
    
    for target_file in missing_files:
        file_path = target_file[0]
        if file_path in deployment_map:
            print(f"   📁 {file_path}")
            print(f"      Source: {deployment_map[file_path]}")
    
    # Step 3: Verification
    print(f"\n3️⃣  Verify Deployment:")
    print("   python deploy_and_verify.py")
    print("   python run_tests.py")

def quick_fix_attempt():
    """Attempt to fix common issues automatically."""
    print(f"\n🔧 Attempting Quick Fixes")
    print("=" * 30)
    
    # Create basic directory structure
    directories = ['src', 'tests', 'logs', 'data_cache']
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"📁 Directory exists: {dir_name}")
    
    # Create __init__.py files
    init_files = ['src/__init__.py', 'tests/__init__.py']
    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.touch()
            print(f"✅ Created: {init_file}")
        else:
            print(f"📄 Exists: {init_file}")
    
    # Check if we have source files in wrong locations
    wrong_location_files = []
    for file_path in Path('.').glob('*.py'):
        if file_path.name.startswith('test_') and 'src' not in str(file_path):
            # Test file in wrong location
            wrong_location_files.append((file_path, f"tests/{file_path.name}"))
        elif file_path.name in ['model_architecture.py', 'data_generation.py', 'logging_utils.py', 
                               'checkpoint_manager.py', 'memorization_metrics.py', 'training_loop.py']:
            # Source file in wrong location  
            wrong_location_files.append((file_path, f"src/{file_path.name}"))
    
    if wrong_location_files:
        print(f"\n📝 Files in wrong locations:")
        for current, target in wrong_location_files:
            print(f"   {current} → {target}")
        print(f"   (Move these files to fix import issues)")

def test_imports():
    """Test if imports work after deployment."""
    print(f"\n🧪 Testing Module Imports")
    print("=" * 30)
    
    # Add src to path
    src_path = Path('src')
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    
    test_imports = [
        ('model_architecture', ['create_gpt_model', 'GPTModel']),
        ('data_generation', ['generate_random_binary_sequences']),
        ('memorization_metrics', ['calculate_morris_memorization']),
        ('training_loop', ['create_training_config']),
        ('logging_utils', ['setup_logging_directories']),
        ('checkpoint_manager', ['save_experiment_state'])
    ]
    
    import_results = []
    
    for module_name, test_attrs in test_imports:
        try:
            module = __import__(module_name)
            
            # Test specific attributes
            missing_attrs = []
            for attr in test_attrs:
                if not hasattr(module, attr):
                    missing_attrs.append(attr)
            
            if missing_attrs:
                import_results.append((module_name, f"⚠️  Missing: {missing_attrs}"))
            else:
                import_results.append((module_name, "✅"))
                
        except ImportError as e:
            import_results.append((module_name, f"❌ Import error: {e}"))
        except Exception as e:
            import_results.append((module_name, f"❌ Error: {e}"))
    
    for module_name, status in import_results:
        print(f"   {module_name}: {status}")
    
    # Overall status
    successful_imports = sum(1 for _, status in import_results if status == "✅")
    print(f"\n📊 Import Success: {successful_imports}/{len(import_results)}")
    
    return successful_imports == len(import_results)

def main():
    """Main deployment verification workflow."""
    print("🚀 Morris Validation Project - Deployment Verification")
    print("=" * 60)
    
    # Step 1: Check file structure
    missing_files = check_file_structure()
    
    # Step 2: Locate existing files
    found_sources = identify_file_locations()
    
    # Step 3: Quick fix attempt
    quick_fix_attempt()
    
    # Step 4: Test imports
    imports_work = test_imports()
    
    # Step 5: Generate instructions
    create_deployment_instructions(missing_files, found_sources)
    
    # Final status
    print(f"\n🎯 Deployment Status")
    print("=" * 25)
    
    if not missing_files and imports_work:
        print("✅ Project fully deployed and working!")
        print("   Ready to run: python run_tests.py")
    elif not missing_files:
        print("⚠️  Files present but imports failing")
        print("   Check file contents and locations")
    else:
        print("❌ Deployment incomplete")
        print(f"   Missing {len(missing_files)} required files")
        print("   Follow deployment instructions above")

if __name__ == "__main__":
    main()

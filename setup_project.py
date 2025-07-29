# setup_project.py
# Script to create the Morris validation project directory structure

from pathlib import Path

def setup_project_structure():
    """Create the complete project directory structure."""
    
    # Define directory structure
    directories = [
        "src",
        "tests", 
        "logs",
        "logs/experiments",
        "logs/training",
        "logs/checkpoints",
        "logs/metrics",
        "logs/metadata"
    ]
    
    # Create directories
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.touch()
            print(f"Created: {init_file}")
    
    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Save the modules to their respective directories")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run tests: python run_tests.py")

if __name__ == "__main__":
    setup_project_structure()

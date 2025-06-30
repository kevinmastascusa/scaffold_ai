"""
Test runner script for Scaffold AI.
Executes all test modules and reports results.
"""

import sys
import os
import logging
from pathlib import Path
import importlib
import traceback

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_test_module(module_path):
    """Run a single test module and return success status."""
    try:
        # Convert path to module name
        module_name = str(module_path.relative_to(project_root)).replace(os.sep, ".").replace(".py", "")
        print(f"\nRunning test module: {module_name}")
        
        # Import and run the module
        module = importlib.import_module(module_name)
        if hasattr(module, 'main'):
            result = module.main()
            return result == 0
        else:
            print("✗ No main() function found in module")
            return False
            
    except Exception as e:
        print(f"✗ Error running test module: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main test runner function."""
    print_separator("Scaffold AI Test Runner")
    
    # Find all test modules
    test_dir = project_root / "scaffold_core" / "scripts" / "tests"
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print("No test files found!")
        return 1
        
    print(f"Found {len(test_files)} test modules:")
    for test_file in test_files:
        print(f"- {test_file.name}")
    
    # Run all test modules
    results = []
    for test_file in test_files:
        success = run_test_module(test_file)
        results.append((test_file.name, success))
    
    # Print summary
    print_separator("Test Results")
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    print(f"Tests completed: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}\n")
    
    if failed > 0:
        print("Failed tests:")
        for name, success in results:
            if not success:
                print(f"- {name}")
        return 1
        
    print("All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
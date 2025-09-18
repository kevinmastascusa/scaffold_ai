#!/usr/bin/env python3
"""
Script to load environment variables and run all tests with the HuggingFace token.
"""

import os
import sys
import subprocess
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    print("🔑 Loading environment variables from .env file...")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value
                print(f"✅ Loaded: {key}")
    
    return True

def verify_token():
    """Verify the HuggingFace token is loaded."""
    token = os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        print("❌ HUGGINGFACE_TOKEN not found in environment")
        return False
    
    print(f"✅ HUGGINGFACE_TOKEN loaded: {token[:10]}...")
    return True

def run_test(test_name, test_file):
    """Run a test and save results."""
    print(f"\n📝 Running {test_name}...")
    
    output_file = f"{test_file.replace('.py', '')}_results_with_token.txt"
    
    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, env=os.environ)
        
        # Save results
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {test_name} completed successfully")
            print(f"📁 Results saved to: {output_file}")
        else:
            print(f"❌ {test_name} failed (return code: {result.returncode})")
            print(f"📁 Error details saved to: {output_file}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running {test_name}: {e}")
        return False

def main():
    """Main function to run all tests."""
    print("🧪 Running Tests with HuggingFace Token")
    print("=" * 50)
    
    # Load environment variables
    if not load_env_file():
        return 1
    
    # Verify token
    if not verify_token():
        return 1
    
    print("\n🧪 Running all tests...")
    print("=" * 50)
    
    # Define tests to run
    tests = [
        ("All Candidates Test", "test_all_candidates.py"),
        ("Prompt Improvements Test", "test_prompt_improvements.py"),
        ("Temperature Config Test", "test_temperature_config.py")
    ]
    
    results = []
    for test_name, test_file in tests:
        success = run_test(test_name, test_file)
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Check the result files for details.")
        return 1

if __name__ == "__main__":
    exit(main()) 
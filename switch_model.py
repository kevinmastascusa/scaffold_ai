#!/usr/bin/env python3
"""
Model Switcher for Scaffold AI
This script allows you to easily switch between different LLM models,
including ONNX-optimized versions for better performance.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_available_models():
    """Get all available models from the config."""
    try:
        from scaffold_core.config import LLM_MODELS
        return LLM_MODELS
    except ImportError:
        logger.error("Failed to import LLM_MODELS from scaffold_core.config")
        return {}

def switch_model(model_key):
    """Switch to the specified model by updating config.py."""
    try:
        # Load the current config
        config_path = Path("scaffold_core/config.py")
        with open(config_path, "r") as f:
            config_content = f.read()
        
        # Get model info
        from scaffold_core.config import LLM_MODELS
        if model_key not in LLM_MODELS:
            logger.error(f"Model key '{model_key}' not found in LLM_MODELS")
            return False
        
        model_name = LLM_MODELS[model_key]["name"]
        
        # Update the SELECTED_LLM_MODEL line
        import re
        new_content = re.sub(
            r'SELECTED_LLM_MODEL = LLM_MODELS\["[^"]+"\]\["name"\]',
            f'SELECTED_LLM_MODEL = LLM_MODELS["{model_key}"]["name"]',
            config_content
        )
        
        # Write the updated config
        with open(config_path, "w") as f:
            f.write(new_content)
        
        logger.info(f"Successfully switched to model: {model_key} ({model_name})")
        
        # Check if model has ONNX flag
        use_onnx = LLM_MODELS[model_key].get("use_onnx", False)
        if use_onnx:
            logger.info("This model will use ONNX Runtime for optimized performance")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        return False

def main():
    """Main function to parse arguments and switch models."""
    # Get available models
    available_models = get_available_models()
    model_keys = list(available_models.keys())
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Switch between different LLM models")
    parser.add_argument(
        "model",
        choices=model_keys,
        help="Model to switch to"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    
    # Parse arguments
    if "--list" in sys.argv:
        print("\nAvailable Models:")
        print("-" * 50)
        for key, info in available_models.items():
            onnx_flag = " [ONNX]" if info.get("use_onnx", False) else ""
            print(f"{key}{onnx_flag}: {info['desc']}")
        print("\nUsage: python switch_model.py MODEL_KEY")
        return
    
    if len(sys.argv) < 2:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    # Switch model
    success = switch_model(args.model)
    
    if success:
        print(f"\nModel switched to: {args.model}")
        print(f"Description: {available_models[args.model]['desc']}")
        print("\nYou can now run your tests or applications with the new model.")
    else:
        print("\nFailed to switch model. See error logs for details.")

if __name__ == "__main__":
    main()
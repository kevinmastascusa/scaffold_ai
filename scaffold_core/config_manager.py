"""
Configuration Manager for Scaffold AI
Handles dynamic model switching and configuration updates.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the configuration manager."""
        if config_file is None:
            # Default to a config file in the project root
            project_root = Path(__file__).parent.parent
            config_file = project_root / "model_config.json"
        
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Default configuration
        return {
            "selected_models": {
                "llm": "tinyllama",
                "embedding": "miniLM"
            },
            "model_settings": {
                "llm": {
                    "temperature": 0.3,
                    "max_new_tokens": 2048,
                    "top_p": 0.9
                },
                "embedding": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }
            }
        }
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")
            raise
    
    def get_selected_model(self, model_type: str) -> Optional[str]:
        """Get the currently selected model for a given type."""
        return self.config.get("selected_models", {}).get(model_type)
    
    def set_selected_model(self, model_type: str, model_key: str) -> bool:
        """Set the selected model for a given type."""
        if "selected_models" not in self.config:
            self.config["selected_models"] = {}
        
        self.config["selected_models"][model_type] = model_key
        self._save_config()
        return True
    
    def get_model_settings(self, model_type: str) -> Dict[str, Any]:
        """Get settings for a specific model type."""
        return self.config.get("model_settings", {}).get(model_type, {})
    
    def update_model_settings(self, model_type: str, settings: Dict[str, Any]) -> bool:
        """Update settings for a specific model type."""
        if "model_settings" not in self.config:
            self.config["model_settings"] = {}
        
        if model_type not in self.config["model_settings"]:
            self.config["model_settings"][model_type] = {}
        
        self.config["model_settings"][model_type].update(settings)
        self._save_config()
        return True
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the full configuration."""
        return self.config.copy()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = {
            "selected_models": {
                "llm": "tinyllama",
                "embedding": "miniLM"
            },
            "model_settings": {
                "llm": {
                    "temperature": 0.3,
                    "max_new_tokens": 2048,
                    "top_p": 0.9
                },
                "embedding": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }
            }
        }
        self._save_config()

# Global instance
config_manager = ConfigManager() 
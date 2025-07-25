#!/usr/bin/env python3
"""
Standalone script to clear conversation memory without requiring full system initialization.
This avoids the HuggingFace token requirement for simple memory clearing operations.
"""

import logging
from pathlib import Path
import json

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_external_conversation_files():
    """Clear all external conversation files."""
    try:
        conversations_dir = Path("conversations")
        if not conversations_dir.exists():
            logger.info("No conversations directory found")
            return 0
        
        cleared_files = 0
        for conversation_file in conversations_dir.glob("*.json"):
            try:
                conversation_file.unlink()
                cleared_files += 1
                logger.info(f"Cleared external file: {conversation_file.name}")
            except Exception as e:
                logger.error(f"Could not delete {conversation_file.name}: {e}")
        
        if cleared_files > 0:
            logger.info(f"Successfully cleared {cleared_files} external conversation files")
        else:
            logger.info("No external conversation files found to clear")
        
        return cleared_files
        
    except Exception as e:
        logger.error(f"Error clearing external conversation files: {e}")
        return 0

def clear_specific_session_file(session_id: str):
    """Clear a specific session's external file."""
    try:
        if not session_id:
            logger.warning("No session_id provided")
            return False
        
        conversations_dir = Path("conversations")
        conversation_file = conversations_dir / f"{session_id}.json"
        
        if conversation_file.exists():
            conversation_file.unlink()
            logger.info(f"Successfully cleared external file for session: {session_id}")
            return True
        else:
            logger.info(f"No external file found for session: {session_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error clearing external file for session {session_id}: {e}")
        return False

def main():
    """Main function to clear all conversation memory."""
    logger.info("Starting conversation memory clearing...")
    
    # Clear all external files
    cleared_count = clear_external_conversation_files()
    
    if cleared_count > 0:
        logger.info(f"✅ Successfully cleared {cleared_count} conversation files")
    else:
        logger.info("ℹ️ No conversation memory found to clear")
    
    logger.info("Memory clearing operation completed")

if __name__ == "__main__":
    main() 
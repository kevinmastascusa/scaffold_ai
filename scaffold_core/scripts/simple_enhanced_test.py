"""
Simple test for enhanced query system
"""

import sys
import os
import warnings

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

warnings.filterwarnings("ignore")  # Suppress all warnings (including FutureWarning)

try:
    import colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    ))
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)

print("Testing enhanced query system...")

try:
    from scaffold_core.vector.enhanced_query import query_enhanced
    print("✅ Successfully imported enhanced query system")
    
    # Test a simple query
    query = "What is life cycle assessment?"
    print(f"\nTesting query: {query}")
    
    result = query_enhanced(query)
    
    print(f"✅ Query completed successfully!")
    print(f"Response: {result['response'][:200]}...")
    print(f"Found {len(result['candidates'])} candidates")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc() 
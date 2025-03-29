"""
Agentic RAG Tutorial Generator application.
"""
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Initializing application")

# Version
__version__ = "0.1.0"

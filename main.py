import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories
def ensure_directories():
    """Ensure all required directories exist."""
    data_dirs = [
        "data/processed",
        "data/embeddings",
        "data/outputs",
        "data/temp"
    ]
    
    for directory in data_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def main():
    """Main entry point for the application."""
    # Import here to avoid circular imports
    from app.config import Config
    from app.models.ollama_client import OllamaClient
    from app.rag.pipeline import RAGPipeline
    from app.tools.tool_registry import ToolRegistry
    from app.agents.agent_manager import AgentManager
    from app.ui.main_interface import launch_ui
    
    # Ensure required directories exist
    ensure_directories()
    
    # Load configuration
    config = Config()
    logger.info("Configuration loaded")
    
    # Initialize model provider
    model_provider = OllamaClient(config.ollama_api_url)
    logger.info(f"Ollama client initialized at {config.ollama_api_url}")
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(config, model_provider)
    logger.info("RAG pipeline initialized")
    
    # Initialize tools
    tool_registry = ToolRegistry(config, model_provider, rag_pipeline)
    logger.info("Tool registry initialized")
    
    # Initialize agent manager
    agent_manager = AgentManager(config, model_provider, rag_pipeline, tool_registry)
    logger.info("Agent manager initialized")
    
    # Launch UI
    logger.info("Launching Streamlit UI")
    
    launch_ui(config, model_provider, rag_pipeline, tool_registry, agent_manager)

if __name__ == "__main__":
    
    main()

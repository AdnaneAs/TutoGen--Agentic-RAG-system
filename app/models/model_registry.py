"""
Registry for tracking and managing available AI models.
"""
import logging
from typing import Dict, List, Optional, Any

from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for tracking and managing available AI models."""
    
    def __init__(self, ollama_client: OllamaClient):
        """Initialize model registry.
        
        Args:
            ollama_client (OllamaClient): Client for interacting with Ollama
        """
        self.ollama_client = ollama_client
        self.models_cache: Dict[str, Dict[str, Any]] = {}
        self.refresh_models()
    
    def refresh_models(self) -> None:
        """Refresh the cache of available models."""
        models = self.ollama_client.list_models()
        
        # Clear and update cache
        self.models_cache.clear()
        for model in models:
            self.models_cache[model["name"]] = model
            
        logger.info(f"Refreshed model registry, found {len(self.models_cache)} models")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Optional[Dict[str, Any]]: Model information or None if not found
        """
        return self.models_cache.get(model_name)
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available models, optionally filtered by type.
        
        Args:
            model_type (Optional[str]): Filter by model type (text, vision, etc.)
            
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        if not self.models_cache:
            self.refresh_models()
            
        if model_type is None:
            return list(self.models_cache.values())
        
        # Filter by type (based on model name and tags)
        filtered_models = []
        for model in self.models_cache.values():
            # Check tags
            tags = model.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
                
            # Check model name and tags
            if (model_type.lower() in model["name"].lower() or 
                any(model_type.lower() in tag.lower() for tag in tags)):
                filtered_models.append(model)
        
        return filtered_models
    
    def get_llm_models(self) -> List[Dict[str, Any]]:
        """Get list of text generation models.
        
        Returns:
            List[Dict[str, Any]]: List of LLM models
        """
        # Text models typically don't have special tags, so exclude models with specific tags
        exclude_tags = ["embed", "vision"]
        
        models = []
        for model in self.models_cache.values():
            tags = model.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
                
            if not any(exclude_tag in tag.lower() for tag in tags for exclude_tag in exclude_tags):
                models.append(model)
        
        return models
    
    def get_embedding_models(self) -> List[Dict[str, Any]]:
        """Get list of embedding models.
        
        Returns:
            List[Dict[str, Any]]: List of embedding models
        """
        return self.list_models("embed")
    
    def get_vision_models(self) -> List[Dict[str, Any]]:
        """Get list of vision models.
        
        Returns:
            List[Dict[str, Any]]: List of vision models
        """
        return self.list_models("vision")
    
    def get_default_llm(self) -> str:
        """Get name of recommended default LLM.
        
        Returns:
            str: Model name
        """
        # Preferred models in order
        preferred = ["llama3", "llama2", "mistral", "phi3"]
        
        llm_models = self.get_llm_models()
        
        # Try to find a preferred model
        for preferred_name in preferred:
            for model in llm_models:
                if preferred_name in model["name"].lower():
                    return model["name"]
        
        # Fallback to first available model
        if llm_models:
            return llm_models[0]["name"]
        
        # Last resort fallback
        return "llama3"
    
    def get_default_embedding_model(self) -> str:
        """Get name of recommended default embedding model.
        
        Returns:
            str: Model name
        """
        # Preferred models in order
        preferred = ["nomic-embed", "all-minilm"]
        
        embedding_models = self.get_embedding_models()
        
        # Try to find a preferred model
        for preferred_name in preferred:
            for model in embedding_models:
                if preferred_name in model["name"].lower():
                    return model["name"]
        
        # Fallback to first available model
        if embedding_models:
            return embedding_models[0]["name"]
        
        # Last resort fallback
        return "nomic-embed-text"
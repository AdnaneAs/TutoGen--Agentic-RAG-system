from typing import List, Dict, Any, Optional
import requests
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding

def get_available_text_embedding_models() -> List[str]:
    """
    Get a list of available text embedding models from Ollama.
    
    Returns:
        List of available embedding model names
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Filter for models that are good for embedding
            embedding_models = [
                model["name"] for model in models 
                if "embed" in model["name"].lower() or 
                "nomic" in model["name"].lower() or
                "all-minilm" in model["name"].lower() or
                "e5" in model["name"].lower()
            ]
            return embedding_models
        else:
            return ["nomic-embed-text"]  # Default fallback
    except Exception as e:
        print(f"Error fetching embedding models: {e}")
        return ["nomic-embed-text"]  # Default fallback

def get_text_embedding_model(model_name: str) -> BaseEmbedding:
    """
    Get a text embedding model.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        Embedding model instance
    """
    if "ollama:" in model_name:
        model_name = model_name.split("ollama:")[1]
        
    # Return Ollama embedding model
    return OllamaEmbedding(model_name=model_name)

class CustomTextEmbedding(BaseEmbedding):
    """Custom text embedding class that can be extended with additional functionality."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize custom text embedding.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        self.embed_model = get_text_embedding_model(model_name)
        super().__init__(**kwargs)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding
        """
        return self.embed_model._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of text embeddings
        """
        return self.embed_model._get_text_embeddings(texts)
        
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embed_model._get_query_embedding(query)
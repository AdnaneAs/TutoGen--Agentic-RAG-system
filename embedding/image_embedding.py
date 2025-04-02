from typing import List, Dict, Any, Optional
import requests
import base64
from pathlib import Path
import numpy as np
from llama_index.embeddings.ollama import OllamaEmbedding

def get_available_image_embedding_models() -> List[str]:
    """
    Get a list of available image embedding/VLM models from Ollama.
    
    Returns:
        List of available VLM model names
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Filter for models that can handle images
            image_models = [
                model["name"] for model in models 
                if any(name in model["name"].lower() for name in 
                      ["llava", "vision", "clip", "cogvlm", "image", "bakllava", "moondream"])
            ]
            return image_models
        else:
            return ["llava"]  # Default fallback
    except Exception as e:
        print(f"Error fetching image models: {e}")
        return ["llava"]  # Default fallback

def get_image_embedding_model(model_name: str):
    """
    Get an image embedding model.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        Image embedding model instance
    """
    # For simplicity and compatibility, we'll use OllamaEmbedding directly
    # instead of the custom class that's causing Pydantic issues
    return OllamaEmbedding(model_name=model_name)

# This simpler approach avoids the Pydantic compatibility issues with
# the custom embedding class. If vision-specific embedding is needed,
# it can be implemented at a higher level in the application.
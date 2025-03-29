"""
Ollama client for interacting with locally deployed AI models.
"""
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Callable

from llama_index.core.llms import CustomLLM
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen
from llama_index.llms.ollama import Ollama
from llama_index.core.base.embeddings.base import BaseEmbedding

from llama_index.embeddings.ollama import OllamaEmbedding

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API to access local models."""
    
    def __init__(self, api_url: str = "http://127.0.0.1:11434"):
        """Initialize Ollama client.
        
        Args:
            api_url (str): URL for Ollama API
        """
        self.api_url = api_url
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                logger.info("Successfully connected to Ollama API")
                return True
            else:
                logger.warning(f"Ollama API returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama API: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models from Ollama.
        
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
            logger.warning(f"Failed to list models: {response.status_code}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_llm(self, model_name: Optional[str] = None) -> CustomLLM:
        """Get LLM instance for specified model.
        
        Args:
            model_name (Optional[str]): Model name to use. Defaults to None.
            
        Returns:
            CustomLLM: LlamaIndex compatible LLM
        """
        if model_name is None:
            # Get the first available model if none specified
            models = self.list_models()
            if models:
                model_name = models[0]["name"]
            else:
                model_name = "llama3"  # Default fallback
                
        return Ollama(model=model_name, base_url=self.api_url)
    
    def get_embedding_model(self, model_name: Optional[str] = None) -> BaseEmbedding:
        """Get embedding model instance.
        
        Args:
            model_name (Optional[str]): Model name to use. Defaults to None.
            
        Returns:
            BaseEmbedding: LlamaIndex compatible embedding model
        """
        if model_name is None:
            # Try to find an embedding model
            models = self.list_models()
            for model in models:
                if "embed" in model["name"].lower():
                    model_name = model["name"]
                    break
            
            # If no embedding model found, use a default one
            if model_name is None:
                model_name = "nomic-embed-text"
        
        return OllamaEmbedding(model_name=model_name, base_url=self.api_url)
    
    def generate(self, 
                 prompt: str, 
                 model_name: Optional[str] = None,
                 stream: bool = False,
                 temperature: float = 0.7,
                 max_tokens: int = 2000) -> str:
        """Generate text from model.
        
        Args:
            prompt (str): Input prompt
            model_name (Optional[str]): Model to use. Defaults to None.
            stream (bool): Whether to stream the response. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 0.7.
            max_tokens (int): Maximum tokens to generate. Defaults to 2000.
            
        Returns:
            str: Generated text
        """
        if model_name is None:
            # Get the first available model if none specified
            models = self.list_models()
            if models:
                model_name = models[0]["name"]
            else:
                model_name = "llama3"  # Default fallback
        
        # Prepare request data
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            if not stream:
                response = requests.post(f"{self.api_url}/api/generate", json=data)
                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    logger.error(f"Error generating text: {response.status_code}")
                    return f"Error: {response.status_code}"
            else:
                # Streaming response
                response = requests.post(f"{self.api_url}/api/generate", json=data, stream=True)
                if response.status_code == 200:
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            chunk_text = chunk.get("response", "")
                            full_response += chunk_text
                    return full_response
                else:
                    logger.error(f"Error generating streaming text: {response.status_code}")
                    return f"Error: {response.status_code}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama API: {e}")
            return f"Error: {str(e)}"
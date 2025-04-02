from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import requests
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

def get_available_table_embedding_models() -> List[str]:
    """
    Get a list of available table embedding models from Ollama.
    Tables generally use the same embedding models as text, but with specialized processing.
    
    Returns:
        List of available table embedding model names
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Filter for models that work well with structured data
            table_models = [
                model["name"] for model in models 
                if "embed" in model["name"].lower() or 
                "nomic" in model["name"].lower() or
                "all-minilm" in model["name"].lower() or
                "e5" in model["name"].lower()
            ]
            return table_models
        else:
            return ["nomic-embed-text"]  # Default fallback
    except Exception as e:
        print(f"Error fetching table embedding models: {e}")
        return ["nomic-embed-text"]  # Default fallback

def get_table_embedding_model(model_name: str):
    """
    Get a table embedding model.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        Table embedding model instance
    """
    # For simplicity and compatibility, we'll use OllamaEmbedding directly
    # instead of a custom class that might cause Pydantic issues
    return OllamaEmbedding(model_name=model_name)

class TableEmbedding(BaseEmbedding):
    """
    Custom embedding class for tables and structured data.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize table embedding.
        
        Args:
            model_name: Name of the underlying text embedding model to use
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        # Use a standard text embedding model under the hood
        self.text_embed_model = OllamaEmbedding(model_name=model_name)
        super().__init__(**kwargs)
    
    def _process_table_text(self, table_text: str) -> str:
        """
        Process table text for better embedding.
        
        Args:
            table_text: Table text to process
            
        Returns:
            Processed table text
        """
        # Try to detect if this is tabular data (CSV-like)
        lines = table_text.strip().split('\n')
        if len(lines) > 1 and ',' in lines[0]:
            # This looks like a CSV
            try:
                # Try to parse as CSV
                df = pd.read_csv(pd.StringIO(table_text))
                # Convert to a more structured text representation
                structured_text = "Table with columns: " + ", ".join(df.columns) + "\n\n"
                
                # Add sample rows
                sample_size = min(5, len(df))
                structured_text += "Sample data:\n"
                for i in range(sample_size):
                    row = df.iloc[i]
                    row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                    structured_text += row_text + "\n"
                
                # Add basic statistics
                structured_text += "\nSummary statistics:\n"
                for col in df.select_dtypes(include=[np.number]).columns:
                    structured_text += f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}\n"
                
                return structured_text
            except:
                # If parsing fails, just use the raw text
                return table_text
        
        return table_text
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for table text.
        
        Args:
            text: Table text to embed
            
        Returns:
            Table text embedding
        """
        processed_text = self._process_table_text(text)
        return self.text_embed_model._get_text_embedding(processed_text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple table texts.
        
        Args:
            texts: List of table texts to embed
            
        Returns:
            List of table text embeddings
        """
        processed_texts = [self._process_table_text(text) for text in texts]
        return self.text_embed_model._get_text_embeddings(processed_texts)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query against tabular data.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.text_embed_model._get_query_embedding(query)
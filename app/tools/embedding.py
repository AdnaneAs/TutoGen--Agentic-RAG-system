"""
Tool for embedding text and images.
"""
import logging
import os
from typing import Dict, List, Any, Union, Optional
import base64
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

class EmbeddingTool:
    """Tool for creating embeddings from different content types."""
    
    def __init__(self, config, model_provider):
        """Initialize embedding tool.
        
        Args:
            config: Application configuration
            model_provider: Model provider for accessing embedding models
        """
        self.config = config
        self.model_provider = model_provider
        self.text_embedding_model = model_provider.get_embedding_model()
    
    def embed_text(self, text: str) -> List[float]:
        """Create embedding for text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Text embedding vector
        """
        try:
            # LlamaIndex embed_query already handles chunking if needed
            embedding = self.text_embedding_model.get_text_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            # Return zeros vector as fallback
            return [0.0] * 384  # Common embedding size
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_image(self, image_path: str) -> Optional[List[float]]:
        """Create embedding for image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Optional[List[float]]: Image embedding vector or None on failure
        """
        # This would typically use a multimodal/vision model
        # For now, we'll use a simple placeholder approach
        try:
            # Check if a vision model is available
            vision_models = self.model_provider.list_models("vision")
            if not vision_models:
                logger.warning("No vision models available for image embedding")
                return None
            
            # Load the image
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Convert image to base64 for API
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Use a method that extracts features from the image
            # This is a simplified implementation
            # In a real system, you would use the vision model's API
            
            # For now, return a random vector as a placeholder
            # In a production system, you would use a proper vision model
            np.random.seed(hash(image_bytes) % 2**32)
            return list(np.random.random(384))  # Common embedding size
            
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return None
    
    def embed_document_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create embeddings for all content in a processed document.
        
        Args:
            content (Dict[str, Any]): Processed document content
            
        Returns:
            Dict[str, Any]: Document content with embeddings added
        """
        # Create a copy to avoid modifying the original
        result = {
            "doc_id": content["doc_id"],
            "path": content["path"],
            "pages": content["pages"],
            "metadata": content["metadata"],
            "text": [],
            "images": [],
            "tables": [],
            "toc": content["toc"]
        }
        
        # Embed text blocks
        for text_block in content["text"]:
            embedded_block = text_block.copy()
            embedded_block["embedding"] = self.embed_text(text_block["content"])
            result["text"].append(embedded_block)
        
        # Embed images
        for image in content["images"]:
            embedded_image = image.copy()
            image_embedding = self.embed_image(image["path"])
            if image_embedding:
                embedded_image["embedding"] = image_embedding
            
            # If image has OCR text, embed that too
            if image.get("ocr_text") and image["ocr_text"].strip():
                embedded_image["ocr_embedding"] = self.embed_text(image["ocr_text"])
                
            result["images"].append(embedded_image)
        
        # Embed tables (using their markdown representation)
        for table in content["tables"]:
            embedded_table = table.copy()
            embedded_table["embedding"] = self.embed_text(table["markdown"])
            result["tables"].append(embedded_table)
        
        logger.info(f"Created embeddings for document {content['doc_id']}: "
                   f"{len(result['text'])} text blocks, {len(result['images'])} images, "
                   f"{len(result['tables'])} tables")
        
        return result
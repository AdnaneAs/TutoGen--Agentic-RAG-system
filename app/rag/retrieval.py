"""
Context retrieval for RAG pipeline.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ContextRetriever:
    """Handles context retrieval from indexed documents."""
    
    def __init__(self, config, rag_pipeline):
        """Initialize context retriever.
        
        Args:
            config: Application configuration
            rag_pipeline: RAG pipeline instance
        """
        self.config = config
        self.rag_pipeline = rag_pipeline
    
    def retrieve_context(self, 
                        query: str, 
                        collection_name: Optional[str] = None,
                        top_k: int = 5) -> Dict[str, Any]:
        """Retrieve context for a query.
        
        Args:
            query (str): Query text
            collection_name (Optional[str]): Name of collection to query
            top_k (int): Number of results to retrieve
            
        Returns:
            Dict[str, Any]: Retrieved context
        """
        # Query the RAG pipeline
        results = self.rag_pipeline.query(query, collection_name, top_k)
        
        # Group results by type
        text_context = []
        images = []
        tables = []
        
        for node in results.get("nodes", []):
            node_type = node.get("type", "text")
            
            if node_type == "text":
                text_context.append({
                    "content": node.get("content", ""),
                    "score": node.get("score", 0),
                    "page": node.get("page"),
                    "doc_id": node.get("doc_id")
                })
            elif node_type == "image":
                images.append({
                    "content": node.get("content", ""),
                    "image_path": node.get("image_path"),
                    "score": node.get("score", 0),
                    "page": node.get("page"),
                    "doc_id": node.get("doc_id")
                })
            elif node_type == "table":
                tables.append({
                    "content": node.get("content", ""),
                    "score": node.get("score", 0),
                    "page": node.get("page"),
                    "doc_id": node.get("doc_id")
                })
        
        return {
            "query": query,
            "collection": collection_name,
            "text_context": text_context,
            "images": images,
            "tables": tables
        }
    
    def retrieve_text_context(self, 
                             query: str, 
                             collection_name: Optional[str] = None,
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve text context for a query.
        
        Args:
            query (str): Query text
            collection_name (Optional[str]): Name of collection to query
            top_k (int): Number of results to retrieve
            
        Returns:
            List[Dict[str, Any]]: Retrieved text context
        """
        # Get all context
        context = self.retrieve_context(query, collection_name, top_k)
        
        # Return just the text context
        return context["text_context"]
    
    def retrieve_images(self, 
                       query: str, 
                       collection_name: Optional[str] = None,
                       top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve images for a query.
        
        Args:
            query (str): Query text
            collection_name (Optional[str]): Name of collection to query
            top_k (int): Number of results to retrieve
            
        Returns:
            List[Dict[str, Any]]: Retrieved images
        """
        # Get all context
        context = self.retrieve_context(query, collection_name, top_k)
        
        # Return just the images
        return context["images"]
    
    def retrieve_tables(self, 
                       query: str, 
                       collection_name: Optional[str] = None,
                       top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve tables for a query.
        
        Args:
            query (str): Query text
            collection_name (Optional[str]): Name of collection to query
            top_k (int): Number of results to retrieve
            
        Returns:
            List[Dict[str, Any]]: Retrieved tables
        """
        # Get all context
        context = self.retrieve_context(query, collection_name, top_k)
        
        # Return just the tables
        return context["tables"]
    
    def retrieve_multimodal_context(self,
                                   query: str,
                                   collection_name: Optional[str] = None,
                                   max_text: int = 5,
                                   max_images: int = 2,
                                   max_tables: int = 2) -> Dict[str, Any]:
        """Retrieve balanced multimodal context.
        
        Args:
            query (str): Query text
            collection_name (Optional[str]): Name of collection to query
            max_text (int): Maximum text items to include
            max_images (int): Maximum images to include
            max_tables (int): Maximum tables to include
            
        Returns:
            Dict[str, Any]: Multimodal context
        """
        # Retrieve a larger set of context
        top_k = max(max_text, max_images, max_tables) * 2
        context = self.retrieve_context(query, collection_name, top_k)
        
        # Limit each type to requested maximum
        limited_context = {
            "query": query,
            "collection": context["collection"],
            "text_context": context["text_context"][:max_text],
            "images": context["images"][:max_images],
            "tables": context["tables"][:max_tables]
        }
        
        return limited_context
    
    def create_prompt_from_context(self, context: Dict[str, Any]) -> str:
        """Create a prompt from retrieved context.
        
        Args:
            context (Dict[str, Any]): Retrieved context
            
        Returns:
            str: Formatted prompt with context
        """
        prompt = f"Query: {context['query']}\n\n"
        prompt += "Context Information:\n\n"
        
        # Add text context
        if context["text_context"]:
            prompt += "Text Passages:\n"
            for i, text in enumerate(context["text_context"], 1):
                prompt += f"Passage {i}:\n{text['content']}\n\n"
        
        # Add image descriptions
        if context["images"]:
            prompt += "Images:\n"
            for i, image in enumerate(context["images"], 1):
                path = image.get("image_path", "")
                desc = image.get("content", "")
                prompt += f"Image {i}: {desc}\n"
                prompt += f"Location: Page {image.get('page')}\n\n"
        
        # Add table descriptions
        if context["tables"]:
            prompt += "Tables:\n"
            for i, table in enumerate(context["tables"], 1):
                prompt += f"Table {i}:\n{table['content']}\n\n"
        
        return prompt
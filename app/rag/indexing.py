"""
Document indexing for RAG pipeline.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from llama_index.core.schema import Document, TextNode, ImageNode

logger = logging.getLogger(__name__)

class DocumentIndexer:
    """Handles document indexing operations."""
    
    def __init__(self, config, rag_pipeline):
        """Initialize document indexer.
        
        Args:
            config: Application configuration
            rag_pipeline: RAG pipeline instance
        """
        self.config = config
        self.rag_pipeline = rag_pipeline
    
    def index_document(self, 
                      doc_id: str,
                      processor_output: Dict[str, Any],
                      collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Index a processed document into the vector store.
        
        Args:
            doc_id (str): Document ID
            processor_output (Dict[str, Any]): Output from PDF processor
            collection_name (Optional[str]): Name for the collection
            
        Returns:
            Dict[str, Any]: Indexing results
        """
        if collection_name is None:
            collection_name = f"doc_{doc_id}"
        
        logger.info(f"Indexing document {doc_id} into collection {collection_name}")
        
        try:
            # Create index
            index = self.rag_pipeline.index_document_content(doc_id, processor_output, collection_name)
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "collection_name": collection_name,
                "text_nodes": len(processor_output.get("text", [])),
                "image_nodes": len(processor_output.get("images", [])),
                "table_nodes": len(processor_output.get("tables", []))
            }
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            return {
                "status": "error",
                "doc_id": doc_id,
                "error": str(e)
            }
    
    def create_nodes_from_content(self, 
                                 doc_id: str, 
                                 content: Dict[str, Any]) -> List[TextNode | ImageNode]:
        """Create LlamaIndex nodes from processed content.
        
        Args:
            doc_id (str): Document ID
            content (Dict[str, Any]): Processed content
            
        Returns:
            List[Union[TextNode, ImageNode]]: List of nodes
        """
        nodes = []
        
        # Create text nodes
        for text_item in content.get("text", []):
            if not text_item["content"].strip():
                continue
                
            node = TextNode(
                text=text_item["content"],
                metadata={
                    "doc_id": doc_id,
                    "page": text_item["page"],
                    "type": "text",
                    "block_id": text_item.get("block_id", f"p{text_item['page']}_text")
                }
            )
            nodes.append(node)
        
        # Create image nodes
        for img_item in content.get("images", []):
            # Use OCR text if available
            ocr_text = img_item.get("ocr_text", "")
            
            # Create image node
            node = ImageNode(
                image_path=img_item["path"],
                text=ocr_text,
                metadata={
                    "doc_id": doc_id,
                    "page": img_item["page"],
                    "type": "image",
                    "image_id": img_item.get("image_id", f"p{img_item['page']}_img")
                }
            )
            nodes.append(node)
        
        # Create nodes for tables
        for table_item in content.get("tables", []):
            node = TextNode(
                text=table_item["markdown"],
                metadata={
                    "doc_id": doc_id,
                    "page": table_item["page"],
                    "type": "table",
                    "table_id": table_item.get("table_id", f"p{table_item['page']}_table"),
                    "rows": table_item.get("rows", 0),
                    "columns": table_item.get("columns", 0)
                }
            )
            nodes.append(node)
        
        return nodes
"""
RAG pipeline for indexing and querying documents.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import ImageNode, TextNode, Document, NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb

logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline for document processing, indexing and retrieval."""
    
    def __init__(self, config, model_provider):
        """Initialize RAG pipeline.
        
        Args:
            config: Application configuration
            model_provider: Model provider for accessing models
        """
        self.config = config
        self.model_provider = model_provider
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=config.vector_db_path)
        self.indexes = {}
    
    def create_service_context(self):
        """Create a service context with current models.
        
        Returns:
            ServiceContext: LlamaIndex service context
        """
        llm = self.model_provider.get_llm()
        embedding_model = self.model_provider.get_embedding_model()
        
        return ServiceContext.from_defaults(
            llm=llm,
            embed_model=embedding_model
        )
    
    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection.
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            chromadb.Collection: ChromaDB collection
        """
        return self.chroma_client.get_or_create_collection(collection_name)
    
    def index_document_content(self, 
                              doc_id: str, 
                              processed_content: Dict[str, Any],
                              collection_name: Optional[str] = None) -> VectorStoreIndex:
        """Index document content into vector store.
        
        Args:
            doc_id (str): Document ID
            processed_content (Dict[str, Any]): Processed document content
            collection_name (Optional[str]): Name of ChromaDB collection
            
        Returns:
            VectorStoreIndex: LlamaIndex vector store index
        """
        if collection_name is None:
            collection_name = f"doc_{doc_id}"
        
        logger.info(f"Indexing document {doc_id} into collection {collection_name}")
        
        # Get or create collection
        collection = self.get_or_create_collection(collection_name)
        
        # Initialize vector store
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create service context
        service_context = self.create_service_context()
        
        # Create nodes from content
        nodes = []
        
        # Text nodes
        for text_item in processed_content["text"]:
            # Skip empty text
            if not text_item["content"].strip():
                continue
                
            node = TextNode(
                text=text_item["content"],
                metadata={
                    "doc_id": doc_id,
                    "page": text_item["page"],
                    "type": "text",
                    "block_id": text_item.get("block_id", f"p{text_item['page']}_text"),
                    "location": str(text_item.get("location", {}))
                }
            )
            nodes.append(node)
        
        # Image nodes
        for img_item in processed_content["images"]:
            # Use OCR text if available, otherwise an empty string
            ocr_text = img_item.get("ocr_text", "")
            
            node = ImageNode(
                image_path=img_item["path"],
                text=ocr_text,
                metadata={
                    "doc_id": doc_id,
                    "page": img_item["page"],
                    "type": "image",
                    "image_id": img_item.get("image_id", f"p{img_item['page']}_img"),
                    "location": str(img_item.get("location", {})),
                    "has_ocr": bool(ocr_text)
                }
            )
            nodes.append(node)
        
        # Table nodes
        for table_item in processed_content["tables"]:
            # Use markdown representation for the text
            table_text = table_item["markdown"]
            
            node = TextNode(
                text=table_text,
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
        
        # Create index from nodes
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            service_context=service_context
        )
        
        # Store reference to index
        self.indexes[collection_name] = index
        
        logger.info(f"Indexed {len(nodes)} nodes for document {doc_id}")
        
        return index
    
    def query(self, 
             query_text: str, 
             collection_name: Optional[str] = None, 
             similarity_top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG index.
        
        Args:
            query_text (str): Query text
            collection_name (Optional[str]): Name of collection to query
            similarity_top_k (int): Number of top results to retrieve
            
        Returns:
            Dict[str, Any]: Query results
        """
        # Get all collections if not specified
        if collection_name is None:
            collections = self.chroma_client.list_collections()
            if not collections:
                return {"error": "No indexed collections found"}
            
            # Query all collections and combine results
            all_results = []
            for collection_info in collections:
                collection_name = collection_info.name
                result = self._query_collection(query_text, collection_name, similarity_top_k)
                if "nodes" in result:
                    all_results.extend(result["nodes"])
            
            # Sort by similarity score
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Limit to top k
            if len(all_results) > similarity_top_k:
                all_results = all_results[:similarity_top_k]
            
            return {
                "query": query_text,
                "nodes": all_results
            }
        else:
            # Query specific collection
            return self._query_collection(query_text, collection_name, similarity_top_k)
    
    def _query_collection(self, 
                         query_text: str, 
                         collection_name: str, 
                         similarity_top_k: int) -> Dict[str, Any]:
        """Query a specific collection.
        
        Args:
            query_text (str): Query text
            collection_name (str): Name of collection to query
            similarity_top_k (int): Number of top results to retrieve
            
        Returns:
            Dict[str, Any]: Query results
        """
        # Get or initialize index
        index = self.indexes.get(collection_name)
        if index is None:
            try:
                # Try to load existing collection
                collection = self.get_or_create_collection(collection_name)
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                service_context = self.create_service_context()
                
                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=storage_context,
                    service_context=service_context
                )
                
                # Cache for future use
                self.indexes[collection_name] = index
            except Exception as e:
                logger.error(f"Error loading index for collection {collection_name}: {e}")
                return {"error": f"Collection not found or error: {str(e)}"}
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k
        )
        
        # Query
        nodes = retriever.retrieve(query_text)
        
        # Process results
        results = []
        for node in nodes:
            node_info = {
                "content": node.node.text,
                "score": node.score,
                "type": node.node.metadata.get("type", "text"),
                "page": node.node.metadata.get("page"),
                "doc_id": node.node.metadata.get("doc_id")
            }
            
            # Add image path for image nodes
            if node_info["type"] == "image" and hasattr(node.node, "image_path"):
                node_info["image_path"] = node.node.image_path
            
            results.append(node_info)
        
        return {
            "query": query_text,
            "collection": collection_name,
            "nodes": results
        }
    
    def generate_response(self, 
                        query_text: str, 
                        collection_name: Optional[str] = None,
                        response_mode: str = "refine") -> Dict[str, Any]:
        """Generate a response using the query engine.
        
        Args:
            query_text (str): Query text
            collection_name (Optional[str]): Name of collection to query
            response_mode (str): Response synthesis mode
            
        Returns:
            Dict[str, Any]: Response and supporting nodes
        """
        # Get all collections if not specified
        if collection_name is None:
            collections = self.chroma_client.list_collections()
            if not collections:
                return {"error": "No indexed collections found"}
            
            # Use first collection as default
            collection_name = collections[0].name
        
        # Get or initialize index
        index = self.indexes.get(collection_name)
        if index is None:
            try:
                # Try to load existing collection
                collection = self.get_or_create_collection(collection_name)
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                service_context = self.create_service_context()
                
                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=storage_context,
                    service_context=service_context
                )
                
                # Cache for future use
                self.indexes[collection_name] = index
            except Exception as e:
                logger.error(f"Error loading index for collection {collection_name}: {e}")
                return {"error": f"Collection not found or error: {str(e)}"}
        
        # Create query engine
        query_engine = index.as_query_engine(
            response_mode=response_mode,
            similarity_top_k=5
        )
        
        # Query
        try:
            response = query_engine.query(query_text)
            
            # Process source nodes
            source_nodes = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    node_info = {
                        "content": node.node.text,
                        "score": node.score,
                        "type": node.node.metadata.get("type", "text"),
                        "page": node.node.metadata.get("page"),
                        "doc_id": node.node.metadata.get("doc_id")
                    }
                    
                    # Add image path for image nodes
                    if node_info["type"] == "image" and hasattr(node.node, "image_path"):
                        node_info["image_path"] = node.node.image_path
                    
                    source_nodes.append(node_info)
            
            return {
                "query": query_text,
                "response": str(response),
                "source_nodes": source_nodes
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": f"Error generating response: {str(e)}"}
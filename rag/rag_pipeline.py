from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import Document, TextNode
from llama_index.core.extractors import BaseExtractor
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core import Settings
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

# Import our custom embeddings
from embedding.text_embedding import get_text_embedding_model
from embedding.image_embedding import get_image_embedding_model
from embedding.tables_embedding import get_table_embedding_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG Pipeline class that implements indexation, querying, embedding and extraction
    using LlamaIndex and ChromaDB.
    """
    
    def __init__(
        self,
        collection_name: str = "default_collection",
        text_embedding_model: str = "nomic-embed-text",
        image_embedding_model: str = "llava",
        table_embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2:latest",
        persist_dir: str = "./chroma_db"
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            collection_name: Name of the ChromaDB collection
            text_embedding_model: Model name for text embedding
            image_embedding_model: Model name for image embedding
            table_embedding_model: Model name for table embedding
            llm_model: LLM model name for generation
            persist_dir: Directory to persist ChromaDB
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.text_embedding_model = text_embedding_model
        self.image_embedding_model = image_embedding_model
        self.table_embedding_model = table_embedding_model
        self.llm_model = llm_model
        self.index = None
        
        # Initialize embedding functions
        self.text_embed_fn = get_text_embedding_model(text_embedding_model)
        self.image_embed_fn = get_image_embedding_model(image_embedding_model)
        self.table_embed_fn = get_table_embedding_model(table_embedding_model)
        
        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            
            try:
                # Get or create collection
                self.collection = self.chroma_client.get_or_create_collection(collection_name)
                print(f"Collection '{collection_name}' ready with {self.collection.count()} items")
                
                # Set up vector store and index
                self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
                self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                
                # Initialize the vector store index if collection has items
                if self.collection.count() > 0:
                    self.index = self.vector_store.to_index()
            except Exception as e:
                print(f"ERROR: Failed to create/get collection: {e}")
                self.collection = None
                self.vector_store = None
                self.storage_context = None
        except Exception as e:
            print(f"ERROR: Failed to initialize ChromaDB client: {e}")
            self.chroma_client = None
            self.collection = None
            self.vector_store = None
            self.storage_context = None
    
    def index_documents(
        self, 
        document_path: str, 
        extractors: Optional[List[BaseExtractor]] = None
    ) -> None:
        """
        Index documents from a directory.
        
        Args:
            document_path: Path to directory containing documents
            extractors: Optional list of extractors to use
        """
        # Load documents from directory
        documents = SimpleDirectoryReader(document_path).load_data()
        
        # Process documents with appropriate embedding models
        processed_documents = []
        
        for doc in documents:
            # Determine document type based on file extension
            file_path = doc.metadata.get("file_path", "")
            
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Image document - use image embedding
                Settings.embed_model = self.image_embed_fn
            elif file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
                # Table document - use table embedding
                Settings.embed_model = self.table_embed_fn
            else:
                # Default to text embedding
                Settings.embed_model = self.text_embed_fn
            
            processed_documents.append(doc)
        
        # Reset to default embedding model
        Settings.embed_model = self.text_embed_fn
        
        # Index documents
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(
                processed_documents,
                storage_context=self.storage_context,
                show_progress=True
            )
        else:
            self.index.insert_nodes([TextNode(text=doc.text, metadata=doc.metadata) for doc in processed_documents])
        
        print(f"Indexed {len(processed_documents)} documents")
    
    def extract_information(
        self, 
        document: Document, 
        extractors: List[BaseExtractor]
    ) -> Dict[str, Any]:
        """
        Extract information from a document using provided extractors.
        
        Args:
            document: Document to extract information from
            extractors: List of extractors to use
            
        Returns:
            Dictionary of extracted information
        """
        extracted_info = {}
        
        for extractor in extractors:
            extraction_result = extractor.extract(document)
            extracted_info[extractor.__class__.__name__] = extraction_result
        
        return extracted_info
    
    def query(
        self, 
        query_text: str, 
        similarity_top_k: int = 3,
        response_mode: str = "compact",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            query_text: Query text
            similarity_top_k: Number of similar documents to retrieve
            response_mode: Response mode (compact, tree, etc.)
            filters: Optional filters for query
            
        Returns:
            Query response
        """
        if self.index is None:
            return {"error": "No documents indexed yet"}
        
        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode,
            filters=self._build_filters(filters) if filters else None
        )
        
        # Query
        response = query_engine.query(query_text)
        
        return {
            "response": str(response),
            "source_nodes": [
                {
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": node.score
                } for node in response.source_nodes
            ]
        }
    
    def _build_filters(self, filter_dict: Dict[str, Any]):
        """
        Convert dictionary of filters to MetadataFilters object.
        
        Args:
            filter_dict: Dictionary of filters
            
        Returns:
            MetadataFilters object
        """
        if not filter_dict:
            return None
            
        filters = []
        for key, value in filter_dict.items():
            filters.append(ExactMatchFilter(key=key, value=value))
            
        return MetadataFilters(filters=filters)
    
    def process_with_llm(self, input_text: str, system_prompt: str = None):
        """
        Process text with the configured LLM model.
        
        Args:
            input_text: Text to process
            system_prompt: Optional system prompt
            
        Returns:
            LLM response
        """
        try:
            # Initialize LLM
            llm = ChatOllama(model=self.llm_model)
            
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
                
            # Add user input
            messages.append(HumanMessage(content=input_text))
            
            # Get response
            response = llm.invoke(messages)
            
            return {
                "text": response.content,
                "model": self.llm_model
            }
        except Exception as e:
            logger.error(f"Error processing with LLM: {str(e)}")
            return {"error": f"Error processing with LLM: {str(e)}"}
    
    def get_collection_stats(self):
        """
        Get statistics about the collection.
        
        Returns:
            Collection statistics
        """
        try:
            if self.collection is None:
                return {"error": "Collection not initialized"}
                
            count = self.collection.count()
            
            # Get content type distribution
            content_types = {}
            try:
                # This requires iterating through items which might not be supported
                # in all ChromaDB versions - using try/except for safety
                for item in self.collection.get()["metadatas"]:
                    if item and "content_type" in item:
                        content_type = item["content_type"]
                        content_types[content_type] = content_types.get(content_type, 0) + 1
            except:
                content_types = {"unknown": count}
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "content_types": content_types,
                "embedding_models": {
                    "text": self.text_embedding_model,
                    "image": self.image_embedding_model,
                    "table": self.table_embedding_model
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": f"Error getting collection stats: {str(e)}"}
    
    def clear_index(self) -> None:
        """
        Clear the index and collection.
        """
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(self.collection_name)
        self.index = None
        print(f"Cleared collection: {self.collection_name}")
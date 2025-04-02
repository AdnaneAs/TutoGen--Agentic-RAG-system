from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import Document
from llama_index.core.extractors import BaseExtractor
from llama_index.core import Settings

# Import our custom embeddings
from embedding.text_embedding import get_text_embedding_model
from embedding.image_embedding import get_image_embedding_model
from embedding.tables_embedding import get_table_embedding_model

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
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
        
        # Get embedding models
        self.text_embedding = get_text_embedding_model(text_embedding_model)
        self.image_embedding = get_image_embedding_model(image_embedding_model)
        self.table_embedding = get_table_embedding_model(table_embedding_model)
        
        # Set up vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        
        # Set up LLM model
        self.llm = self._get_llm_model(llm_model)
        
        # Configure global settings with the text embedding model as default
        Settings.embed_model = self.text_embedding
        Settings.llm = self.llm
        
        # Initialize index if collection has documents
        if len(self.collection.get()["ids"]) > 0:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
        else:
            self.index = None
    
    def _get_llm_model(self, model_name: str):
        """Get the LLM model from Ollama."""
        from llama_index.llms.ollama import Ollama
        return Ollama(model=model_name)
    
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
                Settings.embed_model = self.image_embedding
            elif file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
                # Table document - use table embedding
                Settings.embed_model = self.table_embedding
            else:
                # Default to text embedding
                Settings.embed_model = self.text_embedding
            
            processed_documents.append(doc)
        
        # Reset to default embedding model
        Settings.embed_model = self.text_embedding
        
        # Index documents
        self.index = VectorStoreIndex.from_documents(
            processed_documents,
            vector_store=self.vector_store,
            show_progress=True
        )
        
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
        response_mode: str = "compact"
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            query_text: Query text
            similarity_top_k: Number of similar documents to retrieve
            response_mode: Response mode (compact, tree, etc.)
            
        Returns:
            Query response
        """
        if self.index is None:
            return {"error": "No documents indexed yet"}
        
        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode
        )
        
        # Query
        response = query_engine.query(query_text)
        
        return {
            "response": str(response),
            "source_nodes": [
                {
                    "text": node.node.text,
                    "metadata": node.node.metadata,
                    "score": node.score
                } for node in response.source_nodes
            ]
        }
    
    def clear_index(self) -> None:
        """Clear the index and collection."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(self.collection_name)
        self.index = None
        print(f"Cleared collection: {self.collection_name}")
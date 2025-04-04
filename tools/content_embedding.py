from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import os
import sys
import json
import csv
import glob
from PIL import Image
import numpy as np
import logging
import chromadb
from datetime import datetime

# Add parent directory to path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import RAG pipeline and embedding models
from rag.rag_pipeline import RAGPipeline
from embedding.image_embedding import get_image_embedding_model, embed_image_llava
from embedding.text_embedding import get_text_embedding_model
from embedding.tables_embedding import get_table_embedding_model, TableEmbedding
from llama_index.core.schema import Document, ImageDocument
from llama_index.core import Settings
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_vector(vector, label="Vector", max_items=5):
    if vector is None:
        print(f"DEBUG: {label} is None")
        return
        
    if isinstance(vector, list):
        length = len(vector)
        sample = vector[:max_items]
    elif isinstance(vector, np.ndarray):
        length = vector.shape[0]
        sample = vector[:max_items].tolist()
    else:
        print(f"DEBUG: {label} is not a vector but a {type(vector)}")
        return
        
    print(f"DEBUG: {label} - length: {length}, type: {type(vector).__name__}")
    print(f"DEBUG: {label} - first {len(sample)} values: {sample}")
    print(f"DEBUG: {label} - stats: min={min(vector) if length > 0 else 'N/A'}, " +
          f"max={max(vector) if length > 0 else 'N/A'}, " +
          f"avg={sum(vector)/length if length > 0 else 'N/A'}")

class ContentEmbedder:
    def __init__(
        self,
        collection_name: str = "tutorial_collection",
        text_embedding_model: str = "nomic-embed-text",
        image_embedding_model: str = "llava",
        table_embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2:latest",
        persist_dir: str = "./chroma_db"
    ):
        self.collection_name = collection_name
        self.text_embedding_model = text_embedding_model
        self.image_embedding_model = image_embedding_model
        self.table_embedding_model = table_embedding_model
        self.llm_model = llm_model
        self.persist_dir = persist_dir
        
        print(f"\n{'='*80}")
        print("DEBUG: INITIALIZING CONTENT EMBEDDER")
        print(f"DEBUG: Collection name: {collection_name}")
        print(f"DEBUG: Text embedding model: {text_embedding_model}")
        print(f"DEBUG: Image embedding model: {image_embedding_model}")
        print(f"DEBUG: Table embedding model: {table_embedding_model}")
        print(f"DEBUG: Persist directory: {persist_dir}")
        
        self.rag_pipeline = RAGPipeline(
            collection_name=collection_name,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            table_embedding_model=table_embedding_model,
            llm_model=llm_model,
            persist_dir=persist_dir
        )
        
        try:
            self.text_embed_fn = get_text_embedding_model(text_embedding_model)
            self.image_embed_fn = get_image_embedding_model(image_embedding_model)
            self.table_embed_fn = get_table_embedding_model(table_embedding_model)
            
            print("DEBUG: Successfully initialized embedding models")
            
            try:
                self.chroma_client = chromadb.PersistentClient(path=persist_dir)
                
                try:
                    self.collection = self.chroma_client.get_or_create_collection(collection_name)
                    print(f"DEBUG: Collection '{collection_name}' ready")
                    print(f"DEBUG: Collection has {self.collection.count()} items")
                except Exception as e:
                    print(f"ERROR: Failed to create/get collection: {e}")
            except Exception as e:
                print(f"ERROR: Failed to initialize ChromaDB client: {e}")
                self.chroma_client = None
                self.collection = None
                
        except Exception as e:
            print(f"ERROR: Failed to initialize embedding models: {e}")
            import traceback
            print(traceback.format_exc())
            self.text_embed_fn = None
            self.image_embed_fn = None
            self.table_embed_fn = None
            
        print(f"{'='*80}\n")
        
        self.vlm = ChatOllama(model=image_embedding_model)
    
    def embed_pdf_content(self, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in pdf_content:
            return {"error": f"Cannot embed content: {pdf_content['error']}"}
        
        print(f"\n{'='*80}")
        print("DEBUG: STARTING PDF CONTENT EMBEDDING")
        start_time = datetime.now()
        print(f"DEBUG: PDF filename: {pdf_content.get('filename', 'Unknown')}")
        
        results = {
            "text_embedding": {"status": "pending", "count": 0},
            "table_embedding": {"status": "pending", "count": 0},
            "image_embedding": {"status": "pending", "count": 0}
        }
        
        text_results = self.embed_text_content(pdf_content)
        results["text_embedding"] = text_results
        
        table_results = self.embed_table_content(pdf_content)
        results["table_embedding"] = table_results
        
        image_results = self.process_and_embed_images(pdf_content)
        results["image_embedding"] = image_results
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        print(f"DEBUG: Total processing time: {total_time:.2f} seconds")
        print(f"{'='*80}\n")
        
        return results
    
    def embed_text_content(self, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        try:
            text_documents = []
            
            pdf_filename = pdf_content.get("filename", "unknown")
            doc_id_prefix = pdf_filename.replace(".pdf", "").replace(" ", "_").lower()
            
            for page in pdf_content.get("pages", []):
                page_num = page.get("page_num", 0)
                text = page.get("text", "")
                
                if text.strip():
                    doc = Document(
                        text=text,
                        metadata={
                            "source": pdf_filename,
                            "page_num": page_num,
                            "content_type": "text",
                            "doc_id": f"{doc_id_prefix}_page_{page_num}"
                        }
                    )
                    text_documents.append(doc)
            
            Settings.embed_model = get_text_embedding_model(self.text_embedding_model)
            
            if text_documents:
                if self.rag_pipeline.index is None:
                    self.rag_pipeline.index = self.rag_pipeline.vector_store.to_index()
                
                for doc in text_documents[:1]:  # Debug only the first document as an example
                    if Settings.embed_model:
                        embedding = Settings.embed_model.get_text_embedding(doc.text)
                        debug_vector(embedding, f"Text Embedding (page {doc.metadata['page_num']})")
                
                self.rag_pipeline.index.insert_nodes(text_documents)
                
                return {
                    "status": "success", 
                    "count": len(text_documents),
                    "message": f"Embedded {len(text_documents)} text documents"
                }
            else:
                return {"status": "warning", "count": 0, "message": "No text content found to embed"}
            
        except Exception as e:
            logger.error(f"Error embedding text content: {str(e)}")
            return {"status": "error", "count": 0, "error": str(e)}
    
    def embed_table_content(self, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        try:
            table_documents = []
            
            pdf_filename = pdf_content.get("filename", "unknown")
            doc_id_prefix = pdf_filename.replace(".pdf", "").replace(" ", "_").lower()
            
            for table_idx, table in enumerate(pdf_content.get("tables", [])):
                page_num = table.get("page", 0)
                csv_path = table.get("csv_path", "")
                headers = table.get("headers", [])
                preview = table.get("preview", [])
                
                if csv_path and os.path.exists(csv_path):
                    try:
                        table_text = ""
                        with open(csv_path, 'r', encoding='utf-8') as csv_file:
                            reader = csv.reader(csv_file)
                            for row in reader:
                                table_text += ",".join(row) + "\n"
                        
                        doc = Document(
                            text=table_text,
                            metadata={
                                "source": pdf_filename,
                                "page_num": page_num,
                                "content_type": "table",
                                "headers": ",".join(headers) if headers else "",
                                "table_index": table_idx,
                                "doc_id": f"{doc_id_prefix}_table_{page_num}_{table_idx}"
                            }
                        )
                        table_documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Error processing table CSV {csv_path}: {str(e)}")
            
            Settings.embed_model = get_table_embedding_model(self.table_embedding_model)
            
            if table_documents:
                if self.rag_pipeline.index is None:
                    self.rag_pipeline.index = self.rag_pipeline.vector_store.to_index()
                
                for doc in table_documents[:1]:  # Debug only the first document as an example
                    if Settings.embed_model:
                        embedding = Settings.embed_model.get_text_embedding(doc.text)
                        debug_vector(embedding, f"Table Embedding (page {doc.metadata['page_num']}, table {doc.metadata['table_index']})")
                
                self.rag_pipeline.index.insert_nodes(table_documents)
                
                return {
                    "status": "success", 
                    "count": len(table_documents),
                    "message": f"Embedded {len(table_documents)} table documents"
                }
            else:
                return {"status": "warning", "count": 0, "message": "No table content found to embed"}
            
        except Exception as e:
            logger.error(f"Error embedding table content: {str(e)}")
            return {"status": "error", "count": 0, "error": str(e)}
    
    def process_and_embed_images(self, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_documents = []
            
            pdf_filename = pdf_content.get("filename", "unknown")
            doc_id_prefix = pdf_filename.replace(".pdf", "").replace(" ", "_").lower()
            
            for img_idx, image_info in enumerate(pdf_content.get("images", [])):
                image_path = image_info.get("path", "")
                page_num = image_info.get("page", 0)
                
                if image_path and os.path.exists(image_path):
                    try:
                        logger.info(f"Processing image {img_idx+1}/{len(pdf_content.get('images', []))} from page {page_num}")
                        
                        # Use embed_image_llava to get image description and embedding vector
                        result = embed_image_llava(
                            image_path=image_path,
                            prompt="Describe this image in detail, including any text, diagrams, or visual elements.",
                            model=self.image_embedding_model
                        )
                        
                        if "error" in result:
                            logger.warning(f"Error embedding image {image_path}: {result['error']}")
                            continue
                        
                        # Get the description from the result
                        image_description = result.get("description", "")
                        logger.info(f"Generated description of {len(image_description)} chars")
                        
                        # Create output directory for JSON if it doesn't exist
                        json_dir = os.path.dirname(image_path)
                        os.makedirs(json_dir, exist_ok=True)
                        
                        # Path for JSON metadata
                        json_path = os.path.splitext(image_path)[0] + ".json"
                        
                        # Combine LLaVA result with PDF-specific metadata
                        image_metadata = {
                            "page": page_num,
                            "index": img_idx,
                            "width": image_info.get("width", 0),
                            "height": image_info.get("height", 0),
                            "path": image_path,
                            "description": image_description,
                            "source": pdf_filename
                        }
                        
                        # Add the embedding if available
                        if "embedding" in result:
                            image_metadata["embedding"] = result["embedding"]
                            # Debug the embedding vector using our debug_vector function
                            debug_vector(result["embedding"], f"Image Embedding (page {page_num}, image {img_idx})")
                        
                        # Save the metadata to JSON
                        with open(json_path, 'w', encoding='utf-8') as json_file:
                            json.dump(image_metadata, json_file, indent=2)
                            logger.info(f"Image metadata saved to {json_path}")
                        
                        # Create document for the RAG pipeline
                        doc = Document(
                            text=f"Image Description: {image_description}",
                            metadata={
                                "source": pdf_filename,
                                "page_num": page_num,
                                "content_type": "image",
                                "image_path": image_path,
                                "image_index": img_idx,
                                "doc_id": f"{doc_id_prefix}_image_{page_num}_{img_idx}"
                            }
                        )
                        image_documents.append(doc)
                        
                    except Exception as e:
                        logger.warning(f"Error processing image {image_path}: {str(e)}")
                        import traceback
                        logger.debug(traceback.format_exc())
            
            Settings.embed_model = get_text_embedding_model(self.text_embedding_model)
            
            if image_documents:
                if self.rag_pipeline.index is None:
                    self.rag_pipeline.index = self.rag_pipeline.vector_store.to_index()
                
                self.rag_pipeline.index.insert_nodes(image_documents)
                
                return {
                    "status": "success", 
                    "count": len(image_documents),
                    "message": f"Processed and embedded {len(image_documents)} image descriptions"
                }
            else:
                return {"status": "warning", "count": 0, "message": "No images found to process"}
            
        except Exception as e:
            logger.error(f"Error processing and embedding images: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "count": 0, "error": str(e)}
    
    def generate_image_description(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
                
            import base64
            base64_image = base64.b64encode(image_data).decode("utf-8")
            
            image_prompt = [
                SystemMessage(content="""
                You are an expert image analyst. Given an image, provide a comprehensive detailed description of what you see.
                Your description should include:
                1. The main subject or focus of the image
                2. Important visual elements and their spatial relationships
                3. Any text visible in the image (exactly as written)
                4. Charts, diagrams or visual data representations with detailed descriptions
                5. Style, format and medium (photograph, illustration, chart, etc.)
                6. Context clues that indicate the purpose of the image
                
                Provide a thorough description in 3-5 sentences minimum.
                Focus on objective details rather than subjective interpretations.
                """),
                HumanMessage(content=[
                    {"type": "text", "text": "Please describe this image in detail:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
            
            response = self.vlm.invoke(image_prompt)
            description = response.content
            
            return description
        except Exception as e:
            logger.error(f"Error generating image description: {str(e)}")
            return f"Error generating description: {str(e)}"
import os
import sys
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import numpy as np
import logging
from datetime import datetime
import requests
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding

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

def get_available_text_embedding_models() -> List[str]:
    print("DEBUG: Fetching available text embedding models")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            embedding_models = [
                model["name"] for model in models 
                if any(name in model["name"].lower() for name in 
                      ["embed", "nomic", "e5", "bert", "sentence"])
            ]
            print(f"DEBUG: Found {len(embedding_models)} potential text embedding models: {', '.join(embedding_models)}")
            
            if not embedding_models:
                all_models = [model["name"] for model in models]
                print(f"DEBUG: No specific embedding models found, returning all {len(all_models)} available models")
                return all_models
            return embedding_models
        else:
            print(f"DEBUG: Error fetching models from Ollama API - status code: {response.status_code}")
            return ["nomic-embed-text"]
    except Exception as e:
        print(f"DEBUG: Error fetching text embedding models: {e}")
        return ["nomic-embed-text"]

def get_text_embedding_model(model_name: str = "nomic-embed-text"):
    print(f"DEBUG: Initializing text embedding model: {model_name}")
    
    if not model_name:
        model_name = "nomic-embed-text"
        print(f"DEBUG: Empty model name provided, defaulting to {model_name}")
    
    model_map = {
        "nomic-embed-text": embed_text_nomic,
        "e5": embed_text_e5,
    }
    
    embedding_func = None
    for model_prefix, func in model_map.items():
        if model_prefix.lower() in model_name.lower():
            embedding_func = func
            print(f"DEBUG: Selected {model_prefix} embedding function for {model_name}")
            break
    
    if embedding_func is None:
        print(f"DEBUG: No specific embedding function for {model_name}, using generic Ollama embedding")
        return lambda text: embed_text_ollama(text, model_name)
    
    return embedding_func

def embed_text_nomic(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    print(f"\n{'='*80}")
    print(f"DEBUG: STARTING NOMIC-EMBED-TEXT PROCESSING")
    print(f"DEBUG: Processing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if isinstance(text, str):
        texts = [text]
        single_input = True
        print(f"DEBUG: Processing single text ({len(text)} chars)")
    else:
        texts = text
        single_input = False
        print(f"DEBUG: Processing batch of {len(texts)} texts")
    
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        
        print("DEBUG: Creating Nomic embeddings model")
        embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        
        start_time = datetime.now()
        print(f"DEBUG: Generating embeddings for {len(texts)} text chunks")
        
        embeddings = embeddings_model.embed_documents(texts)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        print(f"DEBUG: Text embedding completed in {processing_time:.2f} seconds")
        
        for i, embedding in enumerate(embeddings):
            text_preview = texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i]
            print(f"\nDEBUG: Embedding {i+1}/{len(embeddings)} for text: '{text_preview}'")
            debug_vector(embedding, f"Text embedding {i+1}")
        
        result = embeddings[0] if single_input else embeddings
        
        print(f"DEBUG: COMPLETED NOMIC-EMBED-TEXT PROCESSING")
        print(f"{'='*80}\n")
        
        return result
    except Exception as e:
        print(f"ERROR: Failed to embed text with nomic-embed-text: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        dim = 768
        if single_input:
            return [0.0] * dim
        else:
            return [[0.0] * dim for _ in texts]

def embed_text_e5(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    print(f"\n{'='*80}")
    print(f"DEBUG: STARTING E5 EMBEDDING PROCESSING")
    print(f"DEBUG: Processing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if isinstance(text, str):
        texts = [text]
        single_input = True
        print(f"DEBUG: Processing single text ({len(text)} chars)")
    else:
        texts = text
        single_input = False
        print(f"DEBUG: Processing batch of {len(texts)} texts")
    
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        
        print("DEBUG: Creating E5 embeddings model")
        embeddings_model = OllamaEmbeddings(model="e5")
        
        start_time = datetime.now()
        print(f"DEBUG: Generating embeddings for {len(texts)} text chunks")
        
        embeddings = embeddings_model.embed_documents(texts)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        print(f"DEBUG: Text embedding completed in {processing_time:.2f} seconds")
        
        for i, embedding in enumerate(embeddings):
            text_preview = texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i]
            print(f"\nDEBUG: Embedding {i+1}/{len(embeddings)} for text: '{text_preview}'")
            debug_vector(embedding, f"E5 embedding {i+1}")
        
        result = embeddings[0] if single_input else embeddings
        
        print(f"DEBUG: COMPLETED E5 EMBEDDING PROCESSING")
        print(f"{'='*80}\n")
        
        return result
    except Exception as e:
        print(f"ERROR: Failed to embed text with E5: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        dim = 1024
        if single_input:
            return [0.0] * dim
        else:
            return [[0.0] * dim for _ in texts]

def embed_text_ollama(text: Union[str, List[str]], model_name: str) -> Union[List[float], List[List[float]]]:
    print(f"\n{'='*80}")
    print(f"DEBUG: STARTING GENERIC OLLAMA EMBEDDING WITH MODEL: {model_name}")
    print(f"DEBUG: Processing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if isinstance(text, str):
        texts = [text]
        single_input = True
        print(f"DEBUG: Processing single text ({len(text)} chars)")
    else:
        texts = text
        single_input = False
        print(f"DEBUG: Processing batch of {len(texts)} texts")
    
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        
        print(f"DEBUG: Creating embeddings with model: {model_name}")
        embeddings_model = OllamaEmbeddings(model=model_name)
        
        start_time = datetime.now()
        print(f"DEBUG: Generating embeddings for {len(texts)} text chunks")
        
        embeddings = embeddings_model.embed_documents(texts)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        print(f"DEBUG: Text embedding completed in {processing_time:.2f} seconds")
        
        for i, embedding in enumerate(embeddings):
            text_preview = texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i]
            print(f"\nDEBUG: Embedding {i+1}/{len(embeddings)} for text: '{text_preview}'")
            debug_vector(embedding, f"Embedding {i+1}")
        
        result = embeddings[0] if single_input else embeddings
        
        print(f"DEBUG: COMPLETED EMBEDDING WITH MODEL: {model_name}")
        print(f"{'='*80}\n")
        
        return result
    except Exception as e:
        print(f"ERROR: Failed to embed text with {model_name}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        dim = 384
        if single_input:
            return [0.0] * dim
        else:
            return [[0.0] * dim for _ in texts]

class CustomTextEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.embed_model = get_text_embedding_model(model_name)
        super().__init__(**kwargs)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        embedding = self.embed_model(text)
        debug_vector(embedding, "Text embedding")
        return embedding
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embed_model(texts)
        for i, embedding in enumerate(embeddings):
            debug_vector(embedding, f"Text embedding {i+1}")
        return embeddings
        
    def _get_query_embedding(self, query: str) -> List[float]:
        embedding = self.embed_model(query)
        debug_vector(embedding, "Query embedding")
        return embedding
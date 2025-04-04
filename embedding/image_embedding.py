import os
import sys
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import numpy as np
import logging
import requests
import base64
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_vector(vector, label="Vector", max_items=5):
    # Helper function to print vector statistics for debugging purposes
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
    
def get_available_image_embedding_models() -> List[str]:
    # Query the Ollama API to get a list of available image-capable models
    print("DEBUG: Fetching available image embedding models from Ollama")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            image_models = [
                model["name"] for model in models 
                if any(name in model["name"].lower() for name in 
                      ["llava", "vision", "clip", "cogvlm", "image", "bakllava", "moondream"])
            ]
            print(f"DEBUG: Found {len(image_models)} image models: {', '.join(image_models)}")
            return image_models
        else:
            print(f"DEBUG: Error fetching models from Ollama API - status code: {response.status_code}")
            return ["llava"]
    except Exception as e:
        print(f"DEBUG: Error fetching image models: {e}")
        return ["llava"]

def get_image_embedding_model(model_name: str = "llava"):
    # Factory function that returns the appropriate embedding function based on model name
    print(f"DEBUG: Initializing image embedding model: {model_name}")
    
    # Handle model names with version tags (e.g., "llava:13b")
    model_base_name = model_name.split(':')[0].lower()
    
    if "llava" in model_base_name:
        def llava_with_specified_model(image_path, prompt="Describe this image in detail."):
            print(f"DEBUG: Using custom LLaVA model: {model_name}")
            return embed_image_llava(image_path, prompt, model=model_name)
        
        return llava_with_specified_model
    else:
        print(f"WARNING: Unknown image embedding model {model_name}. Defaulting to llava.")
        return embed_image_llava

def embed_image_llava(image_path: str, prompt: str = "Describe this image in detail.", model: str = "llava") -> Dict[str, Any]:
    # Process an image with LLaVA and generate an embedding vector
    import base64
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
    
    print(f"\n{'='*80}")
    print(f"DEBUG: STARTING {model} PROCESSING FOR: {image_path}")
    print(f"DEBUG: Processing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DEBUG: Using prompt: '{prompt}'")
    print(f"{'='*80}\n")
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"ERROR: Image file not found: {image_path}")
            return {"error": f"Image file not found: {image_path}"}
            
        # Get image stats
        file_size = os.path.getsize(image_path) / 1024
        print(f"DEBUG: Image file size: {file_size:.2f} KB")
        
        # Get image file name and prepare JSON output path with same name but .json extension
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        image_dir = os.path.dirname(image_path)
        json_path = os.path.join(image_dir, f"{image_name}.json")
        
        print(f"DEBUG: Will save metadata to {json_path}")
        
        # Check for and use existing metadata if available
        if os.path.exists(json_path):
            print(f"DEBUG: Found existing metadata file: {json_path}")
            try:
                with open(json_path, 'r') as f:
                    existing_metadata = json.load(f)
                    print("DEBUG: Existing metadata sample:")
                    print(f"  - Description: {existing_metadata.get('description', 'None')[:100]}...")
                    if 'embedding' in existing_metadata:
                        debug_vector(existing_metadata['embedding'], "Existing embedding")
                    
                if 'description' in existing_metadata and 'embedding' in existing_metadata:
                    print("DEBUG: Reusing existing metadata (already processed)")
                    return existing_metadata
            except Exception as e:
                print(f"DEBUG: Error reading existing metadata: {e}. Will regenerate.")
        
        # Convert image to base64 for API submission
        print("DEBUG: Converting image to base64")
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            print(f"DEBUG: Image converted to base64 (length: {len(base64_image)} chars)")
        
        # Initialize LLaVA model through Ollama
        print(f"DEBUG: Creating LLaVA model: {model}")
        model_instance = ChatOllama(model=model)
        
        # Process the image with the model
        print("DEBUG: Sending image to LLaVA for processing")
        start_time = datetime.now()
        
        # Create a proper HumanMessage with image content
        message_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        human_message = HumanMessage(content=message_content)
        response = model_instance.invoke([human_message])
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        print(f"DEBUG: LLaVA processing completed in {processing_time:.2f} seconds")
        
        # Generate embedding vector from the text description
        print("DEBUG: Generating embedding vector from description")
        
        from hashlib import sha256
        
        def simple_text_embedding(text, vector_size=128):
            # Create a deterministic embedding from text using hashing
            # Note: This is a simple approach for demonstration purposes
            # Production systems should use a proper embedding model
            np.random.seed(hash(text) % 2**32)
            
            hash_obj = sha256(text.encode())
            hash_bytes = hash_obj.digest()
            
            seed_value = int.from_bytes(hash_bytes[:4], byteorder='big')
            np.random.seed(seed_value)
            
            embedding = np.random.uniform(-1, 1, vector_size).tolist()
            return embedding
        
        description_text = response.content
        embedding_vector = simple_text_embedding(description_text)
        
        debug_vector(embedding_vector, "Generated embedding")
        
        # Create metadata with both the description and embedding - use model name as string
        metadata = {
            "description": description_text,
            "embedding": embedding_vector,
            "path": image_path,
            "filename": image_filename,
            "prompt": prompt,
            "model": model,  # Store model name as string, not the model object
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": processing_time
        }
        
        # Save the metadata to disk with the same name as the image
        print(f"DEBUG: Saving metadata to {json_path}")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            print("DEBUG: Metadata successfully saved to disk")
        
        description_preview = description_text[:200] + "..." if len(description_text) > 200 else description_text
        print(f"DEBUG: Image description: {description_preview}")
        
        print(f"\n{'='*80}")
        print(f"DEBUG: COMPLETED LLaVA PROCESSING FOR: {image_path}")
        print(f"DEBUG: Generated {len(embedding_vector)} dimension embedding vector")
        print(f"DEBUG: Generated {len(description_text)} character description")
        print(f"{'='*80}\n")
        
        return metadata
    except Exception as e:
        print(f"ERROR: Failed to process image {image_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

def embed_image_clip(image_path: str) -> Dict[str, Any]:
    # Stub implementation of CLIP embedding (currently returns random vectors)
    # TODO: Implement actual CLIP embedding when needed
    print(f"DEBUG: Processing image with CLIP (stub): {image_path}")
    
    embedding = np.random.rand(512).tolist()
    debug_vector(embedding, "CLIP embedding")
    
    metadata_path = os.path.splitext(image_path)[0] + "_clip_metadata.json"
    metadata = {
        "embedding": embedding,
        "path": image_path,
        "model": "clip",
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        print(f"DEBUG: Saved CLIP metadata to {metadata_path}")
    
    return metadata
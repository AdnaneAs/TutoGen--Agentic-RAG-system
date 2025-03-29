"""
Multimodal content handling for the RAG pipeline.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import io
import base64
import requests

logger = logging.getLogger(__name__)

class MultimodalProcessor:
    """Handles processing of multimodal content (images, tables, etc.)."""
    
    def __init__(self, config, model_provider):
        """Initialize multimodal processor.
        
        Args:
            config: Application configuration
            model_provider: Model provider for accessing models
        """
        self.config = config
        self.model_provider = model_provider
    
    def process_image(self, 
                     image_path: str, 
                     caption: bool = True) -> Dict[str, Any]:
        """Process an image for RAG.
        
        Args:
            image_path (str): Path to image file
            caption (bool): Whether to generate a caption
            
        Returns:
            Dict[str, Any]: Processed image information
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {"error": "Image file not found"}
        
        try:
            # Load image
            img = Image.open(image_path)
            
            # Extract basic information
            width, height = img.size
            format_info = img.format
            mode = img.mode
            
            result = {
                "path": image_path,
                "width": width,
                "height": height,
                "format": format_info,
                "mode": mode
            }
            
            # Generate caption if requested
            if caption:
                try:
                    # Check if a vision model is available
                    vision_models = self.model_provider.list_models("vision")
                    
                    if vision_models:
                        # Use the first available vision model
                        caption_text = self._generate_caption(image_path, vision_models[0]["name"])
                        if caption_text:
                            result["caption"] = caption_text
                except Exception as e:
                    logger.warning(f"Error generating caption: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {"error": str(e)}
    
    def _generate_caption(self, image_path: str, model_name: str) -> Optional[str]:
        """Generate a caption for an image using a vision model.
        
        Args:
            image_path (str): Path to image file
            model_name (str): Name of the vision model
            
        Returns:
            Optional[str]: Generated caption or None on failure
        """
        try:
            # Convert image to base64 for API
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
            
            # For Ollama vision models, we need to format the prompt correctly
            # with the image embedded
            image_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # Create a prompt that asks for a caption
            prompt = "Please provide a brief description of this image. What does it show?"
            
            # Create request to Ollama API
            api_url = f"{self.config.ollama_api_url}/api/generate"
            data = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
            
            response = requests.post(api_url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                caption = result.get("response", "").strip()
                return caption
            else:
                logger.warning(f"Error from Ollama API: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return None
    
    def process_table(self, 
                     table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a table for inclusion in RAG.
        
        Args:
            table_data (Dict[str, Any]): Table data from PDF processor
            
        Returns:
            Dict[str, Any]: Processed table information
        """
        try:
            # Extract key information
            result = {
                "path": table_data.get("path", ""),
                "rows": table_data.get("rows", 0),
                "columns": table_data.get("columns", 0),
                "headers": table_data.get("headers", []),
                "markdown": table_data.get("markdown", "")
            }
            
            # Add a summary of the table's content
            if "headers" in table_data and table_data["headers"]:
                headers_str = ", ".join(table_data["headers"])
                result["summary"] = f"Table with {result['rows']} rows and {result['columns']} columns. Headers: {headers_str}"
            else:
                result["summary"] = f"Table with {result['rows']} rows and {result['columns']} columns."
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing table: {e}")
            return {"error": str(e)}
    
    def create_multimodal_markdown(self, 
                                 content_items: List[Dict[str, Any]]) -> str:
        """Create markdown that includes multimodal content.
        
        Args:
            content_items (List[Dict[str, Any]]): Content items (text, images, tables)
            
        Returns:
            str: Markdown representation
        """
        markdown = ""
        
        for item in content_items:
            item_type = item.get("type", "text")
            
            if item_type == "text":
                # Add text content
                markdown += item.get("content", "") + "\n\n"
            
            elif item_type == "image":
                # Add image with optional caption
                path = item.get("path", "")
                if os.path.exists(path):
                    alt_text = item.get("caption", "Image")
                    rel_path = os.path.basename(path)
                    markdown += f"![{alt_text}]({rel_path})\n\n"
                    
                    # Add caption if available
                    if "caption" in item:
                        markdown += f"*{item['caption']}*\n\n"
            
            elif item_type == "table":
                # Add table markdown
                if "markdown" in item:
                    markdown += item["markdown"] + "\n\n"
                
                # Add caption if available
                if "caption" in item:
                    markdown += f"*{item['caption']}*\n\n"
        
        return markdown
    
    def extract_visual_elements(self, query_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract visual elements from query results.
        
        Args:
            query_results (Dict[str, Any]): Results from RAG query
            
        Returns:
            List[Dict[str, Any]]: List of visual elements (images, tables)
        """
        visual_elements = []
        
        # Process nodes from query results
        for node in query_results.get("nodes", []):
            node_type = node.get("type")
            
            if node_type == "image":
                # Add image
                if "image_path" in node and os.path.exists(node["image_path"]):
                    visual_elements.append({
                        "type": "image",
                        "path": node["image_path"],
                        "score": node.get("score", 0),
                        "page": node.get("page")
                    })
            
            elif node_type == "table":
                # Add table
                content = node.get("content", "")
                if "|" in content:  # Simple check for markdown table
                    visual_elements.append({
                        "type": "table",
                        "markdown": content,
                        "score": node.get("score", 0),
                        "page": node.get("page")
                    })
        
        # Sort by relevance score
        visual_elements.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return visual_elements
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import fitz  # PyMuPDF
import os
import base64
import io
from PIL import Image
import re

class SimplePDFExtractor:
    """Simple tool to extract content from a PDF file including text and images."""
    
    def __init__(self):
        """Initialize the PDF extractor."""
        pass
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted content
        """
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found at {pdf_path}"}
        
        result = {
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "total_pages": 0,
            "pages": [],
            "images": [],
            "metadata": {}
        }
        
        try:
            # Open the PDF
            pdf_document = fitz.open(pdf_path)
            result["total_pages"] = len(pdf_document)
            
            # Extract document metadata
            result["metadata"] = {
                "title": pdf_document.metadata.get("title", ""),
                "author": pdf_document.metadata.get("author", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "keywords": pdf_document.metadata.get("keywords", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "creation_date": str(pdf_document.metadata.get("creationDate", "")),
                "modification_date": str(pdf_document.metadata.get("modDate", "")),
            }
            
            # Extract text and images from each page
            for page_num, page in enumerate(pdf_document):
                # Extract text
                text = page.get_text()
                
                # Extract images
                image_list = page.get_images(full=True)
                page_images = []
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save basic image info
                    image_info = {
                        "page": page_num + 1,
                        "index": img_index,
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "image_type": base_image["ext"],
                        "xref": xref,
                    }
                    
                    # Save image to disk
                    img_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images")
                    os.makedirs(img_dir, exist_ok=True)
                    
                    img_filename = f"page{page_num+1}_img{img_index+1}.{base_image['ext']}"
                    img_path = os.path.join(img_dir, img_filename)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_info["path"] = img_path
                    page_images.append(image_info)
                    result["images"].append(image_info)
                
                # Add page data to result
                result["pages"].append({
                    "page_num": page_num + 1,
                    "text": text,
                    "images": page_images
                })
            
            pdf_document.close()
            return result
            
        except Exception as e:
            return {"error": f"Error extracting PDF content: {str(e)}"}


class SimplePDFSummarizer:
    """Simple tool to summarize PDF content using an LLM."""
    
    def __init__(self, llm_model: str):
        """
        Initialize PDF summarizer.
        
        Args:
            llm_model: Name of the LLM model to use for summarization
        """
        self.llm_model = llm_model
    
    def summarize(self, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize PDF content.
        
        Args:
            pdf_content: Dictionary with PDF content from PDFExtractor
            
        Returns:
            Dictionary with summarized content
        """
        if "error" in pdf_content:
            return {"error": f"Cannot summarize due to extraction error: {pdf_content['error']}"}
        
        # Check if we have pages to summarize
        if not pdf_content.get("pages"):
            return {"error": "No pages found in PDF content"}
        
        from langchain_ollama import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = ChatOllama(model=self.llm_model)
        
        # Create summaries for each page
        page_summaries = []
        for page_data in pdf_content["pages"]:
            page_num = page_data["page_num"]
            page_text = page_data["text"]
            
            # Skip empty pages
            if not page_text.strip():
                page_summaries.append({
                    "page_num": page_num,
                    "summary": "Empty page or no extractable text.",
                    "key_points": []
                })
                continue
            
            # Summarize the page
            response = llm.invoke([
                SystemMessage(content="""
                You are a precise PDF content analyzer. Summarize the provided page text,
                focusing on key information, main ideas, and important details. 
                Also extract key points in a bullet point format.
                
                Return your response in the following JSON format:
                {
                    "summary": "Concise summary of the page content",
                    "key_points": ["Point 1", "Point 2", ...]
                }
                """),
                HumanMessage(content=f"Page {page_num} content: {page_text[:4000]}")
            ])
            
            # Extract JSON from response
            summary_text = response.content
            
            # Extract JSON part from potential non-JSON text
            json_match = re.search(r'{.*}', summary_text, re.DOTALL)
            if json_match:
                summary_text = json_match.group(0)
            
            try:
                import json
                summary_data = json.loads(summary_text)
                summary_data["page_num"] = page_num
                page_summaries.append(summary_data)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                page_summaries.append({
                    "page_num": page_num,
                    "summary": summary_text,
                    "key_points": []
                })
        
        # Create an overall document summary
        all_text = "\n\n".join([page["text"][:1000] for page in pdf_content["pages"]])
        
        response = llm.invoke([
            SystemMessage(content="""
            You are a precise PDF document analyzer. Create a comprehensive summary of the entire document.
            Focus on the main topics, key findings, overall structure, and important conclusions.
            
            Return your response in the following JSON format:
            {
                "document_summary": "Comprehensive summary of the entire document",
                "main_topics": ["Topic 1", "Topic 2", ...],
                "key_findings": ["Finding 1", "Finding 2", ...],
                "document_structure": "Brief description of how the document is organized"
            }
            """),
            HumanMessage(content=f"Document content (first part of each page): {all_text[:6000]}")
        ])
        
        # Extract JSON from response
        overall_summary_text = response.content
        
        # Extract JSON part from potential non-JSON text
        json_match = re.search(r'{.*}', overall_summary_text, re.DOTALL)
        if json_match:
            overall_summary_text = json_match.group(0)
        
        try:
            import json
            overall_summary = json.loads(overall_summary_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            overall_summary = {
                "document_summary": overall_summary_text,
                "main_topics": [],
                "key_findings": [],
                "document_structure": "Could not determine structure"
            }
        
        # Combine results
        result = {
            "filename": pdf_content["filename"],
            "total_pages": pdf_content["total_pages"],
            "metadata": pdf_content["metadata"],
            "overall_summary": overall_summary,
            "page_summaries": page_summaries,
            "images": [
                {
                    "page": img["page"],
                    "path": img["path"],
                    "width": img["width"],
                    "height": img["height"]
                }
                for img in pdf_content["images"]
            ]
        }
        
        return result
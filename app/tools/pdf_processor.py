"""
Tool for processing PDF documents and extracting content.
"""
import io
import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import fitz  # PyMuPDF
import camelot
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import cv2

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Tool for processing PDF documents and extracting content."""
    
    def __init__(self, config):
        """Initialize PDF processor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.temp_dir = Path(config.temp_dir)
        self.processed_dir = Path(config.processed_dir)
        
        # Ensure directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF and extract content.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            Dict[str, Any]: Extracted content including text, images, and tables
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Generate a unique ID for this document
        doc_id = Path(pdf_path).stem
        doc_dir = self.processed_dir / doc_id
        doc_dir.mkdir(exist_ok=True)
        
        try:
            document = fitz.open(pdf_path)
            
            content = {
                "doc_id": doc_id,
                "path": pdf_path,
                "pages": document.page_count,
                "metadata": self._extract_metadata(document),
                "text": self._extract_text(document, doc_dir),
                "images": self._extract_images(document, doc_dir),
                "tables": self._extract_tables(pdf_path, doc_dir),
                "toc": self._extract_toc(document)
            }
            
            # Save processed content summary
            summary_path = doc_dir / "summary.json"
            import json
            with open(summary_path, "w") as f:
                # Convert content to JSON-serializable format
                serializable_content = {
                    "doc_id": content["doc_id"],
                    "path": content["path"],
                    "pages": content["pages"],
                    "metadata": content["metadata"],
                    "text_items": len(content["text"]),
                    "image_items": len(content["images"]),
                    "table_items": len(content["tables"]),
                    "toc_items": len(content["toc"])
                }
                json.dump(serializable_content, f, indent=2)
            
            logger.info(f"PDF processing complete: {len(content['text'])} text blocks, " 
                       f"{len(content['images'])} images, {len(content['tables'])} tables")
            
            return content
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _extract_metadata(self, document) -> Dict[str, Any]:
        """Extract metadata from PDF document.
        
        Args:
            document: PyMuPDF document
            
        Returns:
            Dict[str, Any]: Document metadata
        """
        metadata = document.metadata
        
        # Convert to standard dict (newer PyMuPDF versions return a custom object)
        return {k: v for k, v in metadata.items()}
    
    def _extract_text(self, document, doc_dir: Path) -> List[Dict[str, Any]]:
        """Extract text content from PDF document.
        
        Args:
            document: PyMuPDF document
            doc_dir (Path): Directory to save extracted content
            
        Returns:
            List[Dict[str, Any]]: List of text blocks with metadata
        """
        text_blocks = []
        
        for page_num, page in enumerate(document):
            # Extract text with block information
            blocks = page.get_text("blocks")
            
            for i, block in enumerate(blocks):
                if block[6] == 0:  # Text blocks have type 0
                    text_content = block[4]
                    
                    # Skip empty blocks
                    if not text_content.strip():
                        continue
                    
                    # Extract location information
                    x0, y0, x1, y1 = block[:4]
                    
                    text_blocks.append({
                        "page": page_num,
                        "block_id": f"p{page_num}_b{i}",
                        "content": text_content,
                        "location": {
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1
                        }
                    })
            
            # Save page text to file
            text_file = doc_dir / f"page_{page_num}.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(page.get_text())
        
        return text_blocks
    
    def _extract_images(self, document, doc_dir: Path) -> List[Dict[str, Any]]:
        """Extract images from PDF document.
        
        Args:
            document: PyMuPDF document
            doc_dir (Path): Directory to save extracted content
            
        Returns:
            List[Dict[str, Any]]: List of images with metadata
        """
        images_dir = doc_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        images = []
        
        for page_num, page in enumerate(document):
            # Get list of images on the page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = document.extract_image(xref)
                    
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Save image to file
                        image_filename = f"page{page_num}_img{img_index}.{image_ext}"
                        image_path = images_dir / image_filename
                        
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # Extract location info from the reference
                        rect = None
                        for item in page.get_drawings():
                            if item.get("xref") == xref:
                                rect = item.get("rect")
                                break
                        
                        location = {}
                        if rect:
                            location = {
                                "x0": rect.x0,
                                "y0": rect.y0,
                                "x1": rect.x1,
                                "y1": rect.y1
                            }
                        
                        # Try to extract any text from the image using OCR
                        ocr_text = ""
                        try:
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            ocr_text = pytesseract.image_to_string(pil_image)
                        except Exception as e:
                            logger.warning(f"OCR failed for image {image_path}: {e}")
                            
                        # Add image info to results
                        images.append({
                            "page": page_num,
                            "image_id": f"p{page_num}_img{img_index}",
                            "path": str(image_path),
                            "location": location,
                            "ocr_text": ocr_text,
                            "format": image_ext,
                            "size": len(image_bytes)
                        })
                except Exception as e:
                    logger.error(f"Error extracting image {img_index} from page {page_num}: {e}")
        
        return images
    
    def _extract_tables(self, pdf_path: str, doc_dir: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF document.
        
        Args:
            pdf_path (str): Path to the PDF file
            doc_dir (Path): Directory to save extracted content
            
        Returns:
            List[Dict[str, Any]]: List of tables with metadata
        """
        tables_dir = doc_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        
        tables = []
        
        try:
            # Get total pages
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
            
            # Process each page
            for page_num in range(1, total_pages + 1):  # Camelot uses 1-based page numbers
                try:
                    # Extract tables from the page
                    page_tables = camelot.read_pdf(
                        pdf_path, 
                        pages=str(page_num),
                        flavor='lattice'  # Try lattice method first
                    )
                    
                    # If no tables found with lattice, try stream
                    if len(page_tables) == 0:
                        page_tables = camelot.read_pdf(
                            pdf_path, 
                            pages=str(page_num),
                            flavor='stream'
                        )
                    
                    # Process each table on the page
                    for table_idx, table in enumerate(page_tables):
                        # Convert to DataFrame
                        df = table.df
                        
                        # Skip empty tables
                        if df.empty or df.shape[0] <= 1 or df.shape[1] <= 1:
                            continue
                        
                        # Save as CSV
                        csv_filename = f"page{page_num-1}_table{table_idx}.csv"
                        csv_path = tables_dir / csv_filename
                        df.to_csv(csv_path, index=False)
                        
                        # Get accuracy score
                        accuracy = table.parsing_report.get('accuracy', 0)
                        
                        # Get basic table information
                        table_info = {
                            "page": page_num - 1,  # Convert to 0-based
                            "table_id": f"p{page_num-1}_t{table_idx}",
                            "path": str(csv_path),
                            "rows": df.shape[0],
                            "columns": df.shape[1],
                            "accuracy": accuracy,
                            "headers": df.iloc[0].tolist() if not df.empty else [],
                            # Convert to markdown for text representation
                            "markdown": df.to_markdown(index=False)
                        }
                        
                        tables.append(table_info)
                        
                except Exception as e:
                    logger.warning(f"Error extracting tables from page {page_num}: {e}")
        except Exception as e:
            logger.error(f"Error extracting tables from PDF {pdf_path}: {e}")
        
        return tables
    
    def _extract_toc(self, document) -> List[Dict[str, Any]]:
        """Extract table of contents from PDF document.
        
        Args:
            document: PyMuPDF document
            
        Returns:
            List[Dict[str, Any]]: Table of contents entries
        """
        toc = []
        try:
            # Get the TOC/outline
            outline = document.get_toc()
            
            # Convert to a more usable format
            for item in outline:
                level, title, page = item[:3]
                toc.append({
                    "level": level,
                    "title": title,
                    "page": page - 1  # Convert to 0-based
                })
                
        except Exception as e:
            logger.warning(f"Error extracting table of contents: {e}")
        
        return toc
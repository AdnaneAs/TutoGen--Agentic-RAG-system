"""
Tool for generating and managing markdown content.
"""
import logging
import os
from pathlib import Path
import re
import base64
from typing import Dict, List, Any, Optional, Union
import shutil

logger = logging.getLogger(__name__)

class MarkdownGenerator:
    """Tool for generating and managing markdown content."""
    
    def __init__(self, config):
        """Initialize markdown generator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_document(self, title: str) -> Dict[str, Any]:
        """Create a new markdown document.
        
        Args:
            title (str): Document title
            
        Returns:
            Dict[str, Any]: Document metadata
        """
        # Create a slug from the title
        slug = self._create_slug(title)
        
        # Create document directory
        doc_dir = self.output_dir / slug
        doc_dir.mkdir(exist_ok=True)
        
        # Create assets directory
        assets_dir = doc_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # Initialize document
        doc_path = doc_dir / f"{slug}.md"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
        
        return {
            "title": title,
            "slug": slug,
            "path": str(doc_path),
            "dir": str(doc_dir),
            "assets_dir": str(assets_dir),
            "sections": {}
        }
    
    def add_section(self, 
                   document: Dict[str, Any], 
                   section_title: str, 
                   content: str,
                   position: Optional[int] = None) -> Dict[str, Any]:
        """Add a section to the document.
        
        Args:
            document (Dict[str, Any]): Document metadata
            section_title (str): Section title
            content (str): Section content
            position (Optional[int]): Position to insert (None for append)
            
        Returns:
            Dict[str, Any]: Updated document metadata
        """
        doc_path = document["path"]
        
        # Read existing content
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_content = f.read()
        
        # Format the new section
        section_content = f"\n## {section_title}\n\n{content}\n"
        
        # Add section to document
        if position is None:
            # Append to end
            updated_content = doc_content + section_content
        else:
            # Insert at position
            sections = re.split(r'(?=\n## )', doc_content)
            if position < 0 or position > len(sections):
                position = len(sections)
            
            sections.insert(position, section_content)
            updated_content = "".join(sections)
        
        # Write updated content
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        
        # Update document metadata
        section_id = self._create_slug(section_title)
        document["sections"][section_id] = {
            "title": section_title,
            "id": section_id
        }
        
        return document
    
    def update_section(self, 
                      document: Dict[str, Any], 
                      section_title: str, 
                      content: str) -> Dict[str, Any]:
        """Update an existing section in the document.
        
        Args:
            document (Dict[str, Any]): Document metadata
            section_title (str): Section title
            content (str): New section content
            
        Returns:
            Dict[str, Any]: Updated document metadata
        """
        doc_path = document["path"]
        
        # Read existing content
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_content = f.read()
        
        # Find and replace section
        section_pattern = re.compile(rf'\n## {re.escape(section_title)}\n\n(.*?)(?=\n## |\Z)', 
                                    re.DOTALL)
        
        match = section_pattern.search(doc_content)
        if match:
            # Update existing section
            updated_content = section_pattern.sub(f"\n## {section_title}\n\n{content}\n", 
                                                doc_content)
        else:
            # Section doesn't exist, add it
            return self.add_section(document, section_title, content)
        
        # Write updated content
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        
        return document
    
    def add_image(self, 
                 document: Dict[str, Any], 
                 image_path: str, 
                 alt_text: str,
                 caption: Optional[str] = None) -> str:
        """Add an image to the document.
        
        Args:
            document (Dict[str, Any]): Document metadata
            image_path (str): Path to image file
            alt_text (str): Alternative text for the image
            caption (Optional[str]): Optional caption for the image
            
        Returns:
            str: Markdown code for including the image
        """
        assets_dir = Path(document["assets_dir"])
        
        # Copy image to assets directory
        image_filename = os.path.basename(image_path)
        dest_path = assets_dir / image_filename
        
        shutil.copy2(image_path, dest_path)
        
        # Create relative path for markdown
        rel_path = f"assets/{image_filename}"
        
        # Create markdown code
        if caption:
            md_code = f"![{alt_text}]({rel_path})\n*{caption}*\n\n"
        else:
            md_code = f"![{alt_text}]({rel_path})\n\n"
        
        return md_code
    
    def add_table(self, 
                 document: Dict[str, Any], 
                 table_data: Union[List[List[str]], str],
                 headers: Optional[List[str]] = None,
                 caption: Optional[str] = None) -> str:
        """Add a table to the document.
        
        Args:
            document (Dict[str, Any]): Document metadata
            table_data (Union[List[List[str]], str]): Table data or markdown table
            headers (Optional[List[str]]): Table headers
            caption (Optional[str]): Optional caption for the table
            
        Returns:
            str: Markdown code for the table
        """
        # If table_data is already markdown, use it directly
        if isinstance(table_data, str) and "|" in table_data:
            table_md = table_data
        else:
            # Convert list data to markdown table
            if headers:
                table_md = "| " + " | ".join(headers) + " |\n"
                table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            else:
                table_md = ""
            
            # Add data rows
            for row in table_data:
                table_md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        # Add caption if provided
        if caption:
            table_md += f"*{caption}*\n\n"
        else:
            table_md += "\n"
        
        return table_md
    
    def finalize_document(self, document: Dict[str, Any]) -> str:
        """Finalize the document and return the file path.
        
        Args:
            document (Dict[str, Any]): Document metadata
            
        Returns:
            str: Path to the finalized markdown file
        """
        # Format the document to clean up any issues
        doc_path = document["path"]
        
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Fix any double line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Finalized document: {doc_path}")
        
        return doc_path
    
    def _create_slug(self, text: str) -> str:
        """Create a URL-friendly slug from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: URL-friendly slug
        """
        # Convert to lowercase
        slug = text.lower()
        
        # Replace non-alphanumeric with hyphens
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        
        return slug
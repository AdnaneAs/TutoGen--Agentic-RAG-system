from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import fitz  # PyMuPDF
import os
import base64
import io
from PIL import Image
import re
import pandas as pd
import logging
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import sys

# Import content embedding
from .content_embedding import ContentEmbedder

# Add parent directory to path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import embedding functions
from embedding.image_embedding import get_image_embedding_model
from embedding.text_embedding import get_text_embedding_model
from embedding.tables_embedding import get_table_embedding_model
from rag.rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import camelot for table extraction
try:
    import camelot
except ImportError:
    logger.warning("Camelot not installed. Table extraction will not be available.")
    camelot = None

class SimplePDFExtractor:
    def __init__(self):
        # Initialize PDF extraction tools
        pass
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        # Extract text, images, and tables from a PDF file
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found at {pdf_path}"}
        
        result = {
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "total_pages": 0,
            "pages": [],
            "images": [],
            "tables": [],
            "metadata": {}
        }
        
        try:
            pdf_document = fitz.open(pdf_path)
            result["total_pages"] = len(pdf_document)
            
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
            
            for page_num, page in enumerate(pdf_document):
                text = page.get_text()
                
                image_list = page.get_images(full=True)
                page_images = []
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image_info = {
                        "page": page_num + 1,
                        "index": img_index,
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "image_type": base_image["ext"],
                        "xref": xref,
                    }
                    
                    img_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images")
                    os.makedirs(img_dir, exist_ok=True)
                    
                    img_filename = f"page{page_num+1}_img{img_index+1}.{base_image['ext']}"
                    img_path = os.path.join(img_dir, img_filename)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_info["path"] = img_path
                    page_images.append(image_info)
                    result["images"].append(image_info)
                
                result["pages"].append({
                    "page_num": page_num + 1,
                    "text": text,
                    "images": page_images
                })
            
            pdf_document.close()
            
            doc_dir = Path(os.path.dirname(pdf_path))
            
            if camelot is not None:
                tables = self.extract_tables(pdf_path, doc_dir)
                result["tables"] = tables
            else:
                logger.warning("Camelot not available. Skipping table extraction.")
                result["tables"] = []
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            return {"error": f"Error extracting PDF content: {str(e)}"}
    
    def extract_tables(self, pdf_path: str, doc_dir: Path) -> List[Dict[str, Any]]:
        # Extract tables from PDF using Camelot and convert to structured data
        tables_dir = doc_dir / "extracted_tables"
        tables_dir.mkdir(exist_ok=True)
        
        tables = []
        
        try:
            ghostscript_available = True
            try:
                import ghostscript
            except (ImportError, ModuleNotFoundError):
                try:
                    import subprocess
                    subprocess.run(["gs", "--version"], capture_output=True, check=True)
                except (FileNotFoundError, subprocess.CalledProcessError):
                    ghostscript_available = False
                    logger.warning("Ghostscript is not installed or not in PATH. Lattice table extraction may fail.")
            
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
            
            for page_num in range(1, total_pages + 1):
                try:
                    logger.info(f"Extracting tables from page {page_num}")
                    
                    page_tables = []
                    if ghostscript_available:
                        try:
                            page_tables = camelot.read_pdf(
                                pdf_path, 
                                pages=str(page_num),
                                flavor='lattice',
                                encoding='utf-8'
                            )
                        except Exception as e:
                            logger.warning(f"Lattice table extraction failed: {e}")
                    
                    if len(page_tables) == 0:
                        page_tables = camelot.read_pdf(
                            pdf_path, 
                            pages=str(page_num),
                            flavor='stream',
                            encoding='utf-8'
                        )
                    
                    for table_idx, table in enumerate(page_tables):
                        df = table.df
                        
                        if df.empty or df.shape[0] <= 1 or df.shape[1] <= 1:
                            continue
                        
                        csv_filename = f"page{page_num-1}_table{table_idx}.csv"
                        csv_path = tables_dir / csv_filename
                        df.to_csv(csv_path, index=False)
                        
                        accuracy = table.parsing_report.get('accuracy', 0)
                        
                        preview_rows = min(3, df.shape[0])
                        table_preview = df.head(preview_rows).to_dict('records')
                        
                        table_info = {
                            "page": page_num - 1,
                            "table_index": table_idx + 1,
                            "table_id": f"p{page_num-1}_t{table_idx}",
                            "csv_path": str(csv_path),
                            "rows": df.shape[0],
                            "columns": df.shape[1],
                            "accuracy": accuracy,
                            "headers": df.iloc[0].tolist() if not df.empty else [],
                            "preview": table_preview,
                            "markdown": df.to_markdown(index=False)
                        }
                        
                        tables.append(table_info)
                        logger.info(f"Extracted table {table_idx+1} from page {page_num}")
                        
                except Exception as e:
                    logger.warning(f"Error extracting tables from page {page_num}: {e}")
                    logger.debug(f"Exception details: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Error extracting tables from PDF {pdf_path}: {e}")
            logger.debug(f"Exception details: {str(e)}", exc_info=True)
        
        if len(tables) == 0:
            try:
                logger.info("Attempting table extraction with alternative settings")
                all_tables = camelot.read_pdf(
                    pdf_path,
                    pages='all',
                    flavor='lattice',
                    line_scale=40,
                    encoding='utf-8'
                )
                
                for table_idx, table in enumerate(all_tables):
                    df = table.df
                    
                    if df.empty or df.shape[0] <= 1 or df.shape[1] <= 1:
                        continue
                    
                    page_num = table.page
                    
                    csv_filename = f"page{int(page_num)-1}_table{table_idx}.csv"
                    csv_path = tables_dir / csv_filename
                    df.to_csv(csv_path, index=False)
                    
                    preview_rows = min(3, df.shape[0])
                    table_preview = df.head(preview_rows).to_dict('records')
                    
                    accuracy = table.parsing_report.get('accuracy', 0)
                    
                    table_info = {
                        "page": int(page_num) - 1,
                        "table_index": table_idx + 1,
                        "table_id": f"p{int(page_num)-1}_t{table_idx}",
                        "csv_path": str(csv_path),
                        "rows": df.shape[0],
                        "columns": df.shape[1],
                        "accuracy": accuracy,
                        "headers": df.iloc[0].tolist() if not df.empty else [],
                        "preview": table_preview,
                        "markdown": df.to_markdown(index=False)
                    }
                    
                    tables.append(table_info)
                    logger.info(f"Extracted table {table_idx+1} from page {page_num}")
            except Exception as e:
                logger.error(f"Error during alternative table extraction: {str(e)}")
                logger.debug(f"Exception details: {str(e)}", exc_info=True)
        
        return tables
    
    def extract_and_embed(
        self,
        pdf_path: str,
        collection_name: str = "tutorial_collection",
        text_embedding_model: str = "nomic-embed-text",
        image_embedding_model: str = "llava",
        table_embedding_model: str = "nomic-embed-text",
        persist_dir: str = "./chroma_db"
    ) -> Dict[str, Any]:
        # Extract PDF content and embed it into the RAG pipeline for retrieval
        logger.info(f"Extracting content from PDF: {pdf_path}")
        pdf_content = self.extract(pdf_path)
        
        if "error" in pdf_content:
            return {"error": f"PDF extraction failed: {pdf_content['error']}"}
        
        logger.info("Initializing content embedder")
        embedder = ContentEmbedder(
            collection_name=collection_name,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            table_embedding_model=table_embedding_model,
            persist_dir=persist_dir
        )
        
        logger.info("Embedding extracted content into ChromaDB")
        embed_results = embedder.embed_pdf_content(pdf_content)
        
        return {
            "extraction": {
                "pdf_path": pdf_path,
                "filename": pdf_content["filename"],
                "total_pages": pdf_content["total_pages"],
                "text_pages": len(pdf_content["pages"]),
                "images": len(pdf_content["images"]),
                "tables": len(pdf_content["tables"])
            },
            "embedding": embed_results
        }


class SimplePDFSummarizer:
    def __init__(self, llm_model: str):
        # Initialize the summarizer with specified LLM model
        self.llm_model = llm_model
    
    def summarize(self, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        # Generate a comprehensive summary of PDF content
        if "error" in pdf_content:
            return {"error": f"Cannot summarize due to extraction error: {pdf_content['error']}"}
        
        if not pdf_content.get("pages"):
            return {"error": "No pages found in PDF content"}
        
        from langchain_ollama import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = ChatOllama(model=self.llm_model)
        
        page_summaries = []
        for page_data in pdf_content["pages"]:
            page_num = page_data["page_num"]
            page_text = page_data["text"]
            
            if not page_text.strip():
                page_summaries.append({
                    "page_num": page_num,
                    "summary": "Empty page or no extractable text.",
                    "key_points": []
                })
                continue
            
            response = llm.invoke([
                SystemMessage(content="""
                You are an expert academic content analyzer with deep domain knowledge across multiple fields.
                
                Analyze the provided page text to extract:
                1. A concise but comprehensive summary capturing the page's primary information and significance
                2. Key points and insights, emphasizing theoretical foundations, methodologies, findings, and implications
                3. Technical terminology and concepts central to understanding the content
                
                Present your analysis as structured JSON with these properties:
                {
                    "summary": "A comprehensive 3-5 sentence summary capturing the page's core content and contextual significance",
                    "key_points": ["Point 1 with complete context", "Point 2 with complete context", ...],
                    "technical_terms": [{"term": "Term 1", "context": "Brief explanation"}, ...]
                }
                
                Your analysis should maintain academic rigor while making the content accessible. Focus on identifying:
                - Core arguments and their supporting evidence
                - Methodological approaches described
                - Theoretical frameworks referenced
                - Key findings and their implications
                - Connections to broader academic contexts
                
                Return ONLY valid JSON without additional commentary.
                """),
                HumanMessage(content=f"Page {page_num} content: {page_text[:5000]}")
            ])
            
            summary_text = response.content
            
            json_match = re.search(r'{.*}', summary_text, re.DOTALL)
            if json_match:
                summary_text = json_match.group(0)
            
            try:
                import json
                summary_data = json.loads(summary_text)
                summary_data["page_num"] = page_num
                page_summaries.append(summary_data)
            except json.JSONDecodeError:
                page_summaries.append({
                    "page_num": page_num,
                    "summary": summary_text,
                    "key_points": []
                })
        
        tables_summary = self._summarize_tables(pdf_content.get("tables", []), llm)
        
        all_text = "\n\n".join([page["text"][:1000] for page in pdf_content["pages"]])
        
        table_info = ""
        if pdf_content.get("tables"):
            table_info = f"The document contains {len(pdf_content['tables'])} tables. "
            table_info += "Table summary: " + tables_summary.get("summary", "")
        
        response = llm.invoke([
            SystemMessage(content="""
            You are an expert academic document analyzer with domain expertise across multiple fields.
            
            Create a comprehensive analysis of the document, focusing on:
            1. The key topics, themes, and arguments presented
            2. The theoretical frameworks and methodologies employed
            3. Main findings, conclusions, and their wider implications
            4. The document's structure, organization, and academic contribution
            5. Conceptual relationships between different sections of the document
            
            Return your analysis as structured JSON with these properties:
            {
                "document_summary": "A substantive 5-7 sentence summary capturing the document's overall content, purpose, methodology, findings, and significance",
                "main_topics": ["Topic 1", "Topic 2", ...],
                "key_findings": ["Finding 1", "Finding 2", ...],
                "theoretical_frameworks": ["Framework 1", "Framework 2", ...],
                "methodologies": ["Methodology 1", "Methodology 2", ...],
                "document_structure": "Detailed description of how the document is organized",
                "knowledge_domains": ["Domain 1", "Domain 2", ...],
                "target_audience": "Identified intended audience",
                "academic_significance": "Analysis of the document's contribution to its field"
            }
            
            Your analysis must be academically rigorous, capturing both the explicit content and implicit scholarly context.
            Consider both the stated and unstated assumptions, methodological strengths and limitations, and the document's 
            positioning within its broader academic discourse.
            
            If the document contains tables, incorporate their information into your holistic analysis.
            
            Return ONLY valid JSON without additional commentary.
            """),
            HumanMessage(content=f"Document content (first part of each page): {all_text[:10000]}\n\n{table_info}")
        ])
        
        overall_summary_text = response.content
        
        json_match = re.search(r'{.*}', overall_summary_text, re.DOTALL)
        if json_match:
            overall_summary_text = json_match.group(0)
        
        try:
            import json
            overall_summary = json.loads(overall_summary_text)
        except json.JSONDecodeError:
            overall_summary = {
                "document_summary": overall_summary_text,
                "main_topics": [],
                "key_findings": [],
                "document_structure": "Could not determine structure"
            }
        
        result = {
            "filename": pdf_content["filename"],
            "total_pages": pdf_content["total_pages"],
            "metadata": pdf_content["metadata"],
            "overall_summary": overall_summary,
            "page_summaries": page_summaries,
            "tables_summary": tables_summary,
            "images": [
                {
                    "page": img["page"],
                    "path": img["path"],
                    "width": img["width"],
                    "height": img["height"]
                }
                for img in pdf_content["images"]
            ],
            "tables": [
                {
                    "page": table["page"],
                    "rows": table["rows"],
                    "columns": table["columns"],
                    "csv_path": table["csv_path"],
                    "markdown": table.get("markdown", "")
                }
                for table in pdf_content["tables"]
            ]
        }
        
        return result
    
    def _summarize_tables(self, tables: List[Dict[str, Any]], llm) -> Dict[str, Any]:
        # Generate a summary of tables extracted from the PDF
        if not tables:
            return {
                "summary": "No tables found in the document.",
                "table_details": []
            }
        
        table_details = []
        for table in tables:
            markdown_table = table.get("markdown", "")
            
            preview_text = markdown_table if markdown_table else "Table preview not available"
            
            table_info = {
                "table_index": table["table_index"],
                "page": table["page"],
                "rows": table["rows"],
                "columns": table["columns"],
                "headers": ", ".join(table.get("headers", [])),
                "preview": preview_text
            }
            
            table_prompt = f"""
            Analyze this table from a PDF document:
            
            Table {table['table_index']} on page {table['page']}
            Rows: {table['rows']}
            Columns: {table['columns']}
            Headers: {', '.join(table.get('headers', []))}
            
            Preview of data:
            {preview_text}
            
            Provide a detailed description of what this table appears to contain and its purpose in the document.
            """
            
            response = llm.invoke([
                SystemMessage(content="""
                You are a PDF table analyzer. Provide a concise description of what the table contains
                and its likely purpose in the document. Keep your response to 2-3 sentences.
                """),
                HumanMessage(content=table_prompt)
            ])
            
            table_info["description"] = response.content
            table_details.append(table_info)
        
        tables_overview_prompt = f"""
        The PDF document contains {len(tables)} tables.
        
        {', '.join([f"Table {t['table_index']} on page {t['page']}: {t['description']}" for t in table_details])}
        
        Provide a detailed summary of the tables in this document and their overall significance.
        """
        
        response = llm.invoke([
            SystemMessage(content="""
            You are a PDF table analyzer. Provide a concise overview of the tables in the document
            and their overall significance or purpose. Keep your response to 3-4 sentences.
            """),
            HumanMessage(content=tables_overview_prompt)
        ])
        
        return {
            "summary": response.content,
            "table_details": table_details
        }
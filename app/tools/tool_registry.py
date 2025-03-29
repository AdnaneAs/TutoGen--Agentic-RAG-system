"""
Registry for managing and accessing agent tools.
"""
import logging
from typing import Dict, List, Any, Optional, Callable

from .pdf_processor import PDFProcessor
from .web_search import WebSearchTool
from .embedding import EmbeddingTool
from .markdown_generator import MarkdownGenerator

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Registry for managing and accessing agent tools."""
    
    def __init__(self, config, model_provider, rag_pipeline):
        """Initialize tool registry.
        
        Args:
            config: Application configuration
            model_provider: Model provider for accessing models
            rag_pipeline: RAG pipeline instance
        """
        self.config = config
        self.model_provider = model_provider
        self.rag_pipeline = rag_pipeline
        
        # Initialize tools
        self.tools: Dict[str, Any] = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize available tools."""
        # PDF processor
        self.tools["pdf_processor"] = PDFProcessor(self.config)
        
        # Web search
        self.tools["web_search"] = WebSearchTool(self.config)
        
        # Embedding
        self.tools["embedding"] = EmbeddingTool(self.config, self.model_provider)
        
        # Markdown generator
        self.tools["markdown_generator"] = MarkdownGenerator(self.config)
        
        logger.info(f"Initialized {len(self.tools)} tools")
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a tool by name.
        
        Args:
            tool_name (str): Name of the tool
            
        Returns:
            Optional[Any]: Tool instance or None if not found
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all available tools.
        
        Returns:
            List[str]: List of tool names
        """
        return list(self.tools.keys())
    
    def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """Execute a tool by name.
        
        Args:
            tool_name (str): Name of the tool
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            Any: Result of the tool execution
        """
        # Special case for pdf_processor
        if tool_name == "analyze_pdf_structure":
            return self._analyze_pdf_structure(*args, **kwargs)
        
        # Special case for web search
        if tool_name == "search_web":
            return self._search_web(*args, **kwargs)
        
        # Handle direct tool calls
        if "." in tool_name:
            tool_parts = tool_name.split(".", 1)
            tool_instance = self.get_tool(tool_parts[0])
            if tool_instance and hasattr(tool_instance, tool_parts[1]):
                method = getattr(tool_instance, tool_parts[1])
                if callable(method):
                    return method(*args, **kwargs)
        
        # Regular tool execution
        tool = self.get_tool(tool_name)
        if tool is None:
            logger.error(f"Tool not found: {tool_name}")
            return {"error": f"Tool not found: {tool_name}"}
        
        # Determine the main method to call
        method = None
        if hasattr(tool, "execute"):
            method = tool.execute
        elif hasattr(tool, "run"):
            method = tool.run
        elif hasattr(tool, "process"):
            method = tool.process
        
        if method and callable(method):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return {"error": f"Error executing tool: {str(e)}"}
        else:
            logger.error(f"No executable method found for tool: {tool_name}")
            return {"error": f"No executable method found for tool: {tool_name}"}
    
    # Specialized tool methods
    
    def _analyze_pdf_structure(self, text_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze PDF structure from extracted text.
        
        Args:
            text_content (List[Dict[str, Any]]): Extracted text blocks
            
        Returns:
            Dict[str, Any]: Analysis of PDF structure
        """
        # Simple analysis of structure based on text blocks
        headings = []
        content_sections = []
        
        # Look for headings based on position, font size, etc.
        for block in text_content:
            content = block["content"].strip()
            
            # Skip empty blocks
            if not content:
                continue
            
            # Heuristic for detecting headings
            lines = content.split("\n")
            if len(lines) <= 2 and len(content) < 100:
                headings.append({
                    "text": content,
                    "page": block["page"],
                    "location": block["location"]
                })
            else:
                content_sections.append({
                    "text": content,
                    "page": block["page"],
                    "length": len(content),
                    "location": block["location"]
                })
        
        return {
            "headings": headings,
            "content_sections": content_sections,
            "structure": self._infer_document_structure(headings, content_sections)
        }
    
    def _infer_document_structure(self, headings, content_sections):
        """Infer document structure from detected headings and content.
        
        Args:
            headings: Detected headings
            content_sections: Content sections
            
        Returns:
            List[Dict]: Hierarchical structure of the document
        """
        # Sort headings by page and vertical position
        sorted_headings = sorted(headings, key=lambda h: (h["page"], h["location"]["y0"]))
        
        # Create sections based on headings
        structure = []
        for i, heading in enumerate(sorted_headings):
            section = {
                "title": heading["text"],
                "page": heading["page"],
                "subsections": []
            }
            
            # Check if this is a subsection
            if i > 0 and structure:
                # Heuristic: If heading is indented or smaller, it might be a subsection
                prev_heading = sorted_headings[i-1]
                if heading["location"]["x0"] > prev_heading["location"]["x0"] + 10:
                    # This appears to be indented, so it's a subsection
                    structure[-1]["subsections"].append(section)
                    continue
            
            structure.append(section)
        
        return structure
    
    def _search_web(self, query, **kwargs):
        """Search the web for information.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            Dict: Search results
        """
        if not self.config.enable_web_search:
            return {"error": "Web search is disabled in configuration"}
        
        web_search = self.get_tool("web_search")
        if web_search is None:
            return {"error": "Web search tool not available"}
        
        # Determine type of search
        search_type = kwargs.get("type", "text")
        max_results = kwargs.get("max_results", self.config.web_search_max_results)
        
        if search_type == "news":
            results = web_search.search_news(query, max_results)
        elif search_type == "images":
            results = web_search.search_images(query, max_results)
        else:
            results = web_search.search(query, max_results)
        
        return {
            "query": query,
            "type": search_type,
            "count": len(results),
            "results": results
        }
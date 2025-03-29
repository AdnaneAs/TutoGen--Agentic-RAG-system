"""
Web search tool using DuckDuckGo.
"""
import logging
from typing import Dict, List, Any, Optional

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Tool for performing web searches using DuckDuckGo."""
    
    def __init__(self, config):
        """Initialize web search tool.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.ddgs = DDGS()
        self.max_results = config.web_search_max_results
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for information on the web.
        
        Args:
            query (str): Search query
            max_results (Optional[int]): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        if max_results is None:
            max_results = self.max_results
        
        logger.info(f"Performing web search: {query}")
        
        try:
            # Perform the search
            results = list(self.ddgs.text(query, max_results=max_results))
            
            # Process results
            processed_results = []
            for result in results:
                processed_results.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "href": result.get("href", ""),
                    "source": "DuckDuckGo"
                })
            
            logger.info(f"Web search complete, found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            return []
    
    def search_news(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for news articles.
        
        Args:
            query (str): Search query
            max_results (Optional[int]): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of news results
        """
        if max_results is None:
            max_results = self.max_results
        
        logger.info(f"Performing news search: {query}")
        
        try:
            # Perform the news search
            results = list(self.ddgs.news(query, max_results=max_results))
            
            # Process results
            processed_results = []
            for result in results:
                processed_results.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "href": result.get("url", ""),
                    "source": result.get("source", ""),
                    "published": result.get("published", ""),
                    "type": "news"
                })
            
            logger.info(f"News search complete, found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during news search: {e}")
            return []
    
    def search_images(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for images.
        
        Args:
            query (str): Search query
            max_results (Optional[int]): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of image results
        """
        if max_results is None:
            max_results = self.max_results
        
        logger.info(f"Performing image search: {query}")
        
        try:
            # Perform the image search
            results = list(self.ddgs.images(query, max_results=max_results))
            
            # Process results
            processed_results = []
            for result in results:
                processed_results.append({
                    "title": result.get("title", ""),
                    "image_url": result.get("image", ""),
                    "thumbnail": result.get("thumbnail", ""),
                    "source": result.get("source", ""),
                    "href": result.get("url", ""),
                    "height": result.get("height", 0),
                    "width": result.get("width", 0),
                    "type": "image"
                })
            
            logger.info(f"Image search complete, found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during image search: {e}")
            return []
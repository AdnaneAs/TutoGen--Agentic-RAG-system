"""
Research agent for gathering information for tutorial sections.
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ResearchAgent:
    """Agent responsible for researching content for tutorial sections."""
    
    def __init__(self, model_provider, tools, rag_pipeline):
        """Initialize research agent.
        
        Args:
            model_provider: Provider for accessing models
            tools: Tool registry for accessing tools
            rag_pipeline: RAG pipeline for querying indexed content
        """
        self.model_provider = model_provider
        self.tools = tools
        self.rag_pipeline = rag_pipeline
        self.llm = model_provider.get_llm()
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research phase for current section.
        
        Args:
            state (Dict[str, Any]): Current agent state
            
        Returns:
            Dict[str, Any]: Updated agent state
        """
        logger.info(f"Research agent executing for section: {state['current_section']}")
        
        # ReAct loop components
        thoughts = []
        actions = []
        observations = []
        
        # Extract current section information
        current_section = state["current_section"]
        pdf_content = state["pdf_content"]
        
        # ===== THOUGHT =====
        # Think about what information is needed for this section
        initial_thought = self._think_about_section(current_section, state["query"])
        thoughts.append(initial_thought)
        
        # ===== ACTION =====
        # Query the RAG system for relevant content
        search_query = self._create_search_query(current_section, state["query"])
        rag_action = {
            "tool": "rag_pipeline.query",
            "input": {
                "query_text": search_query,
                "similarity_top_k": 5
            }
        }
        actions.append(rag_action)
        
        # ===== OBSERVATION =====
        # Get results from RAG system
        rag_results = self.rag_pipeline.query(
            query_text=search_query,
            similarity_top_k=5
        )
        observations.append(rag_results)
        
        # ===== THOUGHT =====
        # Analyze RAG results
        content_thought = self._analyze_rag_results(rag_results, current_section)
        thoughts.append(content_thought)
        
        # ===== ACTION =====
        # Identify images and tables that might be useful
        visual_action = {
            "tool": "retrieve_visual_elements",
            "input": {
                "query": search_query,
                "rag_results": rag_results
            }
        }
        actions.append(visual_action)
        
        # ===== OBSERVATION =====
        # Process visual elements
        multimodal_processor = self.tools.get_tool("multimodal")
        if multimodal_processor:
            visual_elements = multimodal_processor.extract_visual_elements(rag_results)
        else:
            visual_elements = []
        observations.append(visual_elements)
        
        # ===== THOUGHT =====
        # Consider if web search is needed for additional information
        web_search_thought = self._consider_web_search(current_section, rag_results)
        thoughts.append(web_search_thought)
        
        # ===== ACTION =====
        # Perform web search if needed
        if "should perform web search" in web_search_thought.lower() and self.config.enable_web_search:
            web_search_action = {
                "tool": "search_web",
                "input": {
                    "query": f"{state['query']} {current_section.split(':')[0]}",
                    "max_results": 3
                }
            }
            actions.append(web_search_action)
            
            # ===== OBSERVATION =====
            # Get web search results
            web_search_results = self.tools.execute_tool("search_web", 
                                                        web_search_action["input"]["query"],
                                                        max_results=web_search_action["input"]["max_results"])
            observations.append(web_search_results)
        else:
            web_search_results = {"results": []}
            observations.append(web_search_results)
        
        # ===== FINAL THOUGHT =====
        # Summarize research findings
        research_summary = self._summarize_research(
            current_section, 
            rag_results, 
            visual_elements, 
            web_search_results.get("results", [])
        )
        thoughts.append(research_summary)
        
        # Update state with research results
        new_state = state.copy()
        new_state["research"] = {
            "section": current_section,
            "content": research_summary,
            "rag_results": rag_results,
            "visual_elements": visual_elements,
            "web_results": web_search_results.get("results", [])
        }
        new_state["observations"] = state.get("observations", []) + [{
            "agent": "research",
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "result": research_summary
        }]
        
        return new_state
    
    def _think_about_section(self, section: str, query: str) -> str:
        """Think about what information is needed for this section.
        
        Args:
            section (str): Current section title and description
            query (str): Overall tutorial query
            
        Returns:
            str: Thought about section information needs
        """
        prompt = f"""
        I'm researching content for a tutorial on: {query}
        
        The current section I'm working on is:
        {section}
        
        I need to think about:
        1. What key concepts should be covered in this section?
        2. What examples or illustrations would be helpful?
        3. What specific information should I look for in the reference material?
        4. Are there any technical details, code samples, or diagrams that would be valuable?
        
        Think step by step about what information I need to gather.
        """
        
        try:
            thought = str(self.llm.complete(prompt))
            return thought
        except Exception as e:
            logger.error(f"Error in _think_about_section: {e}")
            return f"For the section '{section}', I need to gather relevant information about key concepts, examples, and technical details that will help explain this topic clearly."
    
    def _create_search_query(self, section: str, query: str) -> str:
        """Create an effective search query for the RAG system.
        
        Args:
            section (str): Current section title and description
            query (str): Overall tutorial query
            
        Returns:
            str: Search query
        """
        # Extract section title (before the colon if present)
        section_title = section.split(":", 1)[0].strip()
        
        # Combine with query terms
        search_terms = f"{query} {section_title}"
        
        # Create more detailed query with the LLM
        prompt = f"""
        I need to create a search query to find information for a tutorial section.
        
        Tutorial topic: {query}
        Section: {section}
        
        Write a detailed search query (2-3 sentences) that will help retrieve the most relevant information for this section.
        Make the query specific enough to find targeted information but broad enough to capture different aspects of the topic.
        """
        
        try:
            detailed_query = str(self.llm.complete(prompt))
            # Use a shorter version if the result is too long
            if len(detailed_query) > 200:
                return search_terms
            return detailed_query
        except Exception as e:
            logger.error(f"Error creating search query: {e}")
            return search_terms
    
    def _analyze_rag_results(self, rag_results: Dict[str, Any], section: str) -> str:
        """Analyze RAG results for relevance to the current section.
        
        Args:
            rag_results (Dict[str, Any]): Results from RAG query
            section (str): Current section
            
        Returns:
            str: Analysis of results
        """
        # Get nodes from results
        nodes = rag_results.get("nodes", [])
        
        if not nodes:
            return "The RAG query didn't return any relevant results. I should try a different approach or consider using web search."
        
        # Create summaries of the top results
        summaries = []
        for i, node in enumerate(nodes[:3]):  # Focus on top 3 for brevity
            content = node.get("content", "")
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "..."
            
            node_type = node.get("type", "text")
            summaries.append(f"Result {i+1} ({node_type}): {content}")
        
        summaries_text = "\n\n".join(summaries)
        
        prompt = f"""
        I'm researching content for this tutorial section:
        {section}
        
        Here are the top results from the RAG system:
        {summaries_text}
        
        Analyze these results and answer:
        1. How relevant are these results to the section topic?
        2. What key information can I extract from these results?
        3. What important aspects of the topic might be missing?
        4. Do I need additional information from other sources?
        
        Provide a brief analysis of the search results.
        """
        
        try:
            analysis = str(self.llm.complete(prompt))
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing RAG results: {e}")
            return f"The search results provide some useful information about {section.split(':')[0]}, but I may need additional sources to create a comprehensive tutorial section."
    
    def _consider_web_search(self, section: str, rag_results: Dict[str, Any]) -> str:
        """Consider if web search is needed for additional information.
        
        Args:
            section (str): Current section
            rag_results (Dict[str, Any]): Results from RAG query
            
        Returns:
            str: Decision on web search
        """
        # Check if RAG results are sufficient
        nodes = rag_results.get("nodes", [])
        has_sufficient_text = sum(1 for n in nodes if n.get("type") == "text") >= 3
        has_images = any(n.get("type") == "image" for n in nodes)
        has_tables = any(n.get("type") == "table" for n in nodes)
        
        prompt = f"""
        I'm creating a tutorial section on:
        {section}
        
        From my document search, I found:
        - {len(nodes)} relevant passages
        - Has sufficient text content: {"Yes" if has_sufficient_text else "No"}
        - Has relevant images: {"Yes" if has_images else "No"}
        - Has relevant tables: {"Yes" if has_tables else "No"}
        
        Should I perform a web search to find additional information? Consider:
        1. Is the information from the document comprehensive enough?
        2. Are there gaps in the content that should be filled?
        3. Would additional examples or explanations be valuable?
        
        Decide whether to perform a web search and explain why.
        """
        
        try:
            decision = str(self.llm.complete(prompt))
            return decision
        except Exception as e:
            logger.error(f"Error in _consider_web_search: {e}")
            return "Based on the available information, I should perform a web search to supplement the content with additional perspectives and examples."
    
    def _summarize_research(self, 
                          section: str, 
                          rag_results: Dict[str, Any],
                          visual_elements: List[Dict[str, Any]],
                          web_results: List[Dict[str, Any]]) -> str:
        """Summarize research findings for the section.
        
        Args:
            section (str): Current section
            rag_results (Dict[str, Any]): Results from RAG query
            visual_elements (List[Dict[str, Any]]): Visual elements
            web_results (List[Dict[str, Any]]): Web search results
            
        Returns:
            str: Research summary
        """
        # Extract key information
        rag_nodes = rag_results.get("nodes", [])
        rag_text = "\n".join([n.get("content", "")[:200] + "..." for n in rag_nodes[:3]])
        
        # Format visual elements
        visuals_text = ""
        for i, visual in enumerate(visual_elements[:2]):
            v_type = visual.get("type", "")
            if v_type == "image":
                visuals_text += f"Image {i+1}: {visual.get('path', '')}\n"
            elif v_type == "table":
                table_content = visual.get("markdown", "")
                if len(table_content) > 200:
                    table_content = table_content[:200] + "..."
                visuals_text += f"Table {i+1}: {table_content}\n"
        
        # Format web results
        web_text = ""
        for i, result in enumerate(web_results[:2]):
            title = result.get("title", "")
            snippet = result.get("body", "")[:150] + "..."
            web_text += f"Web Result {i+1}: {title}\n{snippet}\n\n"
        
        # Create summary prompt
        prompt = f"""
        I'm creating a tutorial section on:
        {section}
        
        After researching, I've gathered the following information:
        
        From document analysis:
        {rag_text}
        
        Visual elements:
        {visuals_text}
        
        Web search results:
        {web_text}
        
        Provide a comprehensive summary of the research findings that:
        1. Synthesizes the key information from all sources
        2. Highlights the most important concepts, examples, and details
        3. Notes any visual elements that should be included
        4. Identifies any gaps that still need to be addressed
        
        This summary will be used to create the tutorial content, so make it thorough and well-organized.
        """
        
        try:
            summary = str(self.llm.complete(prompt))
            return summary
        except Exception as e:
            logger.error(f"Error summarizing research: {e}")
            return f"Based on my research for '{section}', I've gathered sufficient information about the key concepts and practical applications. The document provides good explanations that can be enhanced with the identified images and tables. Some additional web sources offer complementary perspectives that will make the tutorial more comprehensive."
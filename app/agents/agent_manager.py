"""
Agent manager for orchestrating the tutorial generation workflow.
"""
import logging
from typing import Dict, List, Any, Optional, Callable

from langgraph.graph import StateGraph, END

from .planning_agent import PlanningAgent
from .research_agent import ResearchAgent
from .content_agent import ContentAgent

logger = logging.getLogger(__name__)

class AgentManager:
    """Manager for orchestrating agents in the tutorial generation workflow."""
    
    def __init__(self, config, model_provider, rag_pipeline, tool_registry):
        """Initialize agent manager.
        
        Args:
            config: Application configuration
            model_provider: Provider for accessing models
            rag_pipeline: RAG pipeline for content retrieval
            tool_registry: Registry of available tools
        """
        self.config = config
        self.model_provider = model_provider
        self.rag_pipeline = rag_pipeline
        self.tool_registry = tool_registry
        
        # Initialize agents
        self.planning_agent = PlanningAgent(model_provider, tool_registry)
        self.research_agent = ResearchAgent(model_provider, tool_registry, rag_pipeline)
        self.content_agent = ContentAgent(model_provider, tool_registry, rag_pipeline)
        
        # Build agent workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow graph.
        
        Returns:
            StateGraph: Agent workflow graph
        """
        # Define the workflow graph
        workflow = StateGraph(Dict)
        
        # Add nodes
        workflow.add_node("planning", self.planning_agent.execute)
        workflow.add_node("research", self.research_agent.execute)
        workflow.add_node("content_generation", self.content_agent.execute)
        
        # Add edges
        workflow.add_edge("planning", "research")
        workflow.add_edge("research", "content_generation")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "content_generation",
            self._next_section_or_end,
            {
                "continue": "research",
                "end": END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("planning")
        
        # Compile workflow
        return workflow.compile()
    
    def _next_section_or_end(self, state: Dict[str, Any]) -> str:
        """Determine if we should continue to next section or end.
        
        Args:
            state (Dict[str, Any]): Current workflow state
            
        Returns:
            str: Next transition ('continue' or 'end')
        """
        # Check if we've completed all sections
        if state.get("completed", False):
            logger.info("All sections completed, ending workflow")
            return "end"
        
        logger.info(f"Moving to next section: {state.get('current_section')}")
        return "continue"
    
    def run(self, query: str, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full agent workflow.
        
        Args:
            query (str): User query/request
            pdf_content (Dict[str, Any]): Processed PDF content
            
        Returns:
            Dict[str, Any]: Final workflow state
        """
        logger.info(f"Starting agent workflow for query: {query}")
        
        # Initialize state
        initial_state = {
            "query": query,
            "pdf_content": pdf_content,
            "plan": [],
            "current_section": "",
            "tutorial_content": {},
            "observations": [],
            "completed": False,
            "config": self.config
        }
        
        # Run workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            logger.info("Agent workflow completed successfully")
            return final_state
        except Exception as e:
            logger.error(f"Error running agent workflow: {e}")
            
            # Return partial state if available
            if "tutorial_content" in initial_state and initial_state["tutorial_content"]:
                return initial_state
            
            # Create error state
            return {
                "query": query,
                "error": str(e),
                "tutorial_content": {
                    "Error": f"An error occurred while generating the tutorial: {str(e)}"
                },
                "completed": True
            }
    
    def run_with_callbacks(self, 
                         query: str, 
                         pdf_content: Dict[str, Any],
                         state_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """Run agent workflow with progress callbacks.
        
        Args:
            query (str): User query/request
            pdf_content (Dict[str, Any]): Processed PDF content
            state_callback (Optional[Callable]): Callback function for state updates
            
        Returns:
            Dict[str, Any]: Final workflow state
        """
        logger.info(f"Starting agent workflow with callbacks for query: {query}")
        
        # Initialize state
        initial_state = {
            "query": query,
            "pdf_content": pdf_content,
            "plan": [],
            "current_section": "",
            "tutorial_content": {},
            "observations": [],
            "completed": False,
            "config": self.config
        }
        
        # Create custom state handler
        if state_callback:
            state_callback(initial_state)
        
        # Define step callback
        def on_step(state: Dict[str, Any]) -> None:
            if state_callback:
                state_callback(state)
        
        # Run workflow with step callbacks
        try:
            # Use thread for async operation if needed
            import threading
            
            # Create result container
            result_container = {"result": None}
            
            # Define thread function
            def run_workflow():
                try:
                    result = self.workflow.invoke(
                        initial_state,
                        {"on_step": on_step}
                    )
                    result_container["result"] = result
                except Exception as e:
                    logger.error(f"Error in workflow thread: {e}")
                    result_container["error"] = str(e)
            
            # Run in thread
            thread = threading.Thread(target=run_workflow)
            thread.start()
            thread.join()
            
            # Get result
            if "error" in result_container:
                raise Exception(result_container["error"])
            
            final_state = result_container["result"]
            logger.info("Agent workflow completed successfully")
            return final_state
            
        except Exception as e:
            logger.error(f"Error running agent workflow: {e}")
            
            # Return partial state if available
            if "tutorial_content" in initial_state and initial_state["tutorial_content"]:
                return initial_state
            
            # Create error state
            error_state = {
                "query": query,
                "error": str(e),
                "tutorial_content": {
                    "Error": f"An error occurred while generating the tutorial: {str(e)}"
                },
                "completed": True
            }
            
            # Notify callback
            if state_callback:
                state_callback(error_state)
            
            return error_state
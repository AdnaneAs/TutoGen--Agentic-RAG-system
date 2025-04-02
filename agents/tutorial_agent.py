from typing import List, Dict, Any, Annotated, TypedDict, Literal, Union, Optional
import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from operator import itemgetter
import os
import json
from pydantic import BaseModel, Field

# Import simplified tools
from tools.pdf_tools import SimplePDFExtractor, SimplePDFSummarizer
from tools.tutorial_tools import (
    SimpleTutorialPlanner, 
    SimpleTutorialWriter, 
    SimpleTutorialFormatter, 
    SimpleTutorialImprover
)

# Import RAG pipeline
from rag.rag_pipeline import RAGPipeline

# Define state for the agent system
class AgentState(TypedDict):
    """State for the tutorial generation agent."""
    # Input
    input: str
    pdf_path: str
    
    # Process tracking
    status: str
    current_step: str
    
    # Knowledge components
    pdf_content: Optional[Dict[str, Any]]
    extracted_knowledge: Optional[Dict[str, Any]]
    
    # Planning components
    tutorial_plan: Optional[List[Dict[str, Any]]]
    
    # Output components
    tutorial_sections: Optional[Dict[str, str]]
    
    # Final result
    final_tutorial: Optional[str]
    improvements: Optional[List[str]]
    
    # Messaging components
    messages: List[Dict[str, Any]]
    
class TutorialAgent:
    """
    Tutorial generation agent using simplified tools to avoid Pydantic issues.
    """
    
    def __init__(
        self,
        llm_model: str = "llama3.2:latest",
        collection_name: str = "tutorial_collection",
        text_embedding_model: str = "nomic-embed-text",
        image_embedding_model: str = "llava"
    ):
        """
        Initialize the tutorial agent.
        
        Args:
            llm_model: Model name for the LLM
            collection_name: Name of the ChromaDB collection
            text_embedding_model: Model name for text embedding
            image_embedding_model: Model name for image embedding
        """
        self.llm_model = llm_model
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline(
            collection_name=collection_name,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            llm_model=llm_model
        )
        
        # Initialize simplified tools - these don't use BaseTool/Pydantic
        self.pdf_extractor = SimplePDFExtractor()
        self.pdf_summarizer = SimplePDFSummarizer(self.llm_model)
        self.tutorial_planner = SimpleTutorialPlanner(self.llm_model)
        self.tutorial_writer = SimpleTutorialWriter(self.llm_model, self.rag_pipeline)
        self.tutorial_formatter = SimpleTutorialFormatter(self.llm_model)
        self.tutorial_improver = SimpleTutorialImprover(self.llm_model)
        
        # Build the agent graph
        self.build_agent_graph()
    
    def build_agent_graph(self):
        """Build the langgraph agent system."""
        from langchain_ollama import ChatOllama
        from langchain.tools import StructuredTool
        
        # Create structured tools that wrap our simplified tools 
        # This avoids Pydantic inheritance issues with BaseTool
        tools = [
            StructuredTool.from_function(
                func=self.extract_pdf,
                name="extract_pdf",
                description="Extract content from a PDF file including text and images."
            ),
            StructuredTool.from_function(
                func=self.summarize_pdf,
                name="summarize_pdf",
                description="Summarize the content extracted from a PDF file."
            ),
            StructuredTool.from_function(
                func=self.plan_tutorial,
                name="plan_tutorial",
                description="Plan the structure of a tutorial based on PDF content."
            ),
            StructuredTool.from_function(
                func=self.write_tutorial,
                name="write_tutorial",
                description="Write individual sections of a tutorial based on the tutorial plan."
            ),
            StructuredTool.from_function(
                func=self.format_tutorial,
                name="format_tutorial",
                description="Format the tutorial with proper markdown, code samples, and examples."
            ),
            StructuredTool.from_function(
                func=self.improve_tutorial,
                name="improve_tutorial",
                description="Identify and implement improvements to the tutorial."
            )
        ]
        
        # Use Ollama for local LLM
        llm = ChatOllama(model=self.llm_model)
        
        # Create the ReAct agent
        react_agent = create_react_agent(llm, tools)
        
        # Define the nodes in our agent graph
        nodes = {
            "agent": react_agent,
            "tools": ToolNode(tools),
        }
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes to the graph
        workflow.add_node("agent", nodes["agent"])
        workflow.add_node("tools", nodes["tools"])
        
        # Define edges
        workflow.add_edge("agent", "tools")
        workflow.add_edge("tools", "agent")
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Define end condition - using conditional edges if available,
        # otherwise use a simpler approach for older LangGraph versions
        try:
            # Check if add_conditional_edges method exists (newer versions)
            if hasattr(workflow, 'add_conditional_edges'):
                def should_end(state: AgentState) -> bool:
                    """Determine if the agent should end processing."""
                    # Check if the latest message contains the END signal
                    messages = state["messages"]
                    if not messages:
                        return False
                    
                    latest_message = messages[-1]
                    # Handle different message formats (dict or AIMessage object)
                    if hasattr(latest_message, 'content'):
                        # AIMessage format
                        message_content = latest_message.content
                    elif isinstance(latest_message, dict) and "content" in latest_message:
                        # Dict format
                        message_content = latest_message.get("content", "")
                    else:
                        # Unknown format
                        return False
                    
                    return message_content and "TUTORIAL GENERATION COMPLETE" in message_content
                
                # Add conditional edge to end
                workflow.add_conditional_edges(
                    "agent",
                    should_end,
                    {True: END, False: "tools"}
                )
            else:
                # Fallback for older versions - use simpler approach
                # This approach doesn't use conditional edges
                # It will rely on the agent to determine when to stop
                print("Using fallback graph approach for older LangGraph version")
        except Exception as e:
            # Just continue with basic graph if there's any issue
            print(f"Using fallback graph approach due to error: {e}")
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def extract_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Tool that calls our simplified PDF extractor."""
        return self.pdf_extractor.extract(pdf_path)
    
    def summarize_pdf(self, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        """Tool that calls our simplified PDF summarizer."""
        return self.pdf_summarizer.summarize(pdf_content)
    
    def plan_tutorial(self, summarized_content: Dict[str, Any], tutorial_goal: str) -> Dict[str, Any]:
        """Tool that calls our simplified tutorial planner."""
        return self.tutorial_planner.plan(summarized_content, tutorial_goal)
    
    def write_tutorial(self, tutorial_plan: Dict[str, Any], pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        """Tool that calls our simplified tutorial writer."""
        return self.tutorial_writer.write(tutorial_plan, pdf_content)
    
    def format_tutorial(self, written_sections: Dict[str, Any], pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        """Tool that calls our simplified tutorial formatter."""
        return self.tutorial_formatter.format(written_sections, pdf_content)
    
    def improve_tutorial(self, formatted_tutorial: Dict[str, Any]) -> Dict[str, Any]:
        """Tool that calls our simplified tutorial improver."""
        return self.tutorial_improver.improve(formatted_tutorial)
    
    def run(self, pdf_path: str, goal: str) -> Dict[str, Any]:
        """
        Run the agent to generate a tutorial from a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            goal: Description of the tutorial goal
            
        Returns:
            Generated tutorial and metadata
        """
        try:
            # Try using the graph-based approach first
            try:
                # Initial state
                initial_state = {
                    "input": goal,
                    "pdf_path": pdf_path,
                    "status": "started",
                    "current_step": "extract_pdf",
                    "pdf_content": None,
                    "extracted_knowledge": None,
                    "tutorial_plan": None,
                    "tutorial_sections": None,
                    "final_tutorial": None,
                    "improvements": None,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""Create a tutorial from the PDF at {pdf_path}. 
                            The goal of the tutorial is: {goal}. 
                            Follow these steps:
                            1. Extract content from the PDF
                            2. Summarize key information
                            3. Plan the tutorial structure
                            4. Write each section of the tutorial
                            5. Format the tutorial with proper markdown, code samples, and examples
                            6. Identify and implement improvements
                            7. Finalize the tutorial
                            
                            When complete, include the text 'TUTORIAL GENERATION COMPLETE' in your final response.
                            """
                        }
                    ]
                }
                
                # Run the graph
                result = self.graph.invoke(initial_state)
                
                # Return the final tutorial and metadata
                return {
                    "tutorial": result.get("final_tutorial", ""),
                    "improvements": result.get("improvements", []),
                    "plan": result.get("tutorial_plan", {}),
                    "extracted_knowledge": result.get("extracted_knowledge", {})
                }
                
            except Exception as graph_error:
                print(f"Graph-based approach failed: {graph_error}. Falling back to sequential approach.")
                # If the graph approach fails, fall back to sequential approach
                
                # Sequential approach as fallback
                print("Starting tutorial generation process (sequential approach)...")
                
                # Step 1: Extract PDF content
                print("Step 1: Extracting PDF content...")
                pdf_content = self.extract_pdf(pdf_path)
                
                # Step 2: Summarize PDF content
                print("Step 2: Summarizing PDF content...")
                pdf_summary = self.summarize_pdf(pdf_content)
                
                # Step 3: Plan tutorial
                print("Step 3: Planning tutorial structure...")
                tutorial_plan = self.plan_tutorial(pdf_summary, goal)
                
                # Step 4: Write tutorial
                print("Step 4: Writing tutorial content...")
                written_sections = self.write_tutorial(tutorial_plan, pdf_content)
                
                # Step 5: Format tutorial
                print("Step 5: Formatting tutorial...")
                formatted_tutorial = self.format_tutorial(written_sections, pdf_content)
                
                # Step 6: Improve tutorial
                print("Step 6: Improving tutorial...")
                improved_tutorial = self.improve_tutorial(formatted_tutorial)
                
                # Return the results
                return {
                    "tutorial": improved_tutorial.get("improved_content", ""),
                    "improvements": improved_tutorial.get("improvements", []),
                    "plan": tutorial_plan,
                    "extracted_knowledge": pdf_summary
                }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in tutorial generation: {e}")
            print(error_trace)
            
            # Try to return a partial result if something went wrong
            partial_result = {}
            
            # Check what variables are defined to build a partial result
            if 'pdf_content' in locals():
                partial_result["pdf_content"] = pdf_content
            
            if 'pdf_summary' in locals():
                partial_result["extracted_knowledge"] = pdf_summary
            
            if 'tutorial_plan' in locals():
                partial_result["plan"] = tutorial_plan
            
            if 'improved_tutorial' in locals() and improved_tutorial.get("improved_content"):
                partial_result["tutorial"] = improved_tutorial.get("improved_content")
            elif 'formatted_tutorial' in locals() and formatted_tutorial.get("formatted_content"):
                partial_result["tutorial"] = formatted_tutorial.get("formatted_content")
            elif 'written_sections' in locals():
                # Combine sections if we have them
                tutorial_text = ""
                for key, value in written_sections.get("sections", {}).items():
                    if isinstance(value, str):
                        tutorial_text += value + "\n\n"
                partial_result["tutorial"] = tutorial_text
            
            # Add error information
            partial_result["error"] = str(e)
            partial_result["error_trace"] = error_trace
            
            return partial_result
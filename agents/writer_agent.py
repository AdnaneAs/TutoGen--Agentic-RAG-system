from typing import List, Dict, Any, Optional, Type
import sys
from pathlib import Path
import os
import json
import re
import logging

# Add parent directory to path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import tools
from tools.writer_tools import AcademicWriterTool
from rag.rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WriterReActAgent:
    """
    A writer agent that follows the ReAct pattern:
    1. Observe: Analyze the current state and tutorial plan
    2. Think: Reason about what section to write or what information to retrieve
    3. Act: Execute a specific writing action (retrieve info, write section)
    4. Repeat until the tutorial is complete
    """
    
    def __init__(
        self,
        llm_model: str = "llama3.2:latest",
        collection_name: str = "tutorial_collection",
        text_embedding_model: str = "nomic-embed-text",
        image_embedding_model: str = "llava",
        table_embedding_model: str = "nomic-embed-text"
    ):
        """
        Initialize the writer agent.
        
        Args:
            llm_model: Name of the LLM model
            collection_name: Name of the ChromaDB collection
            text_embedding_model: Name of the text embedding model
            image_embedding_model: Name of the image embedding model
            table_embedding_model: Name of the table embedding model
        """
        self.llm_model = llm_model
        self.last_result = None  # Track the last result for state management
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline(
            collection_name=collection_name,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            table_embedding_model=table_embedding_model,
            llm_model=llm_model
        )
        
        # Initialize writer tool
        self.writer_tool = AcademicWriterTool(llm_model, self.rag_pipeline)
        
        # Import LLM for reasoning
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(model=self.llm_model)
    
    def run(self, tutorial_plan: Dict[str, Any], pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the writer agent to generate tutorial sections based on a plan.
        
        Args:
            tutorial_plan: Dictionary containing the tutorial plan
            pdf_content: Dictionary containing the extracted PDF content
            
        Returns:
            Dictionary with written sections and metadata
        """
        # Initialize state
        state = {
            "tutorial_plan": tutorial_plan,
            "pdf_content": pdf_content,
            "current_step": "initialize",
            "section_index": -1,  # Start with introduction (-1)
            "sections_to_write": len(tutorial_plan.get("sections", [])) + 2,  # +2 for intro and conclusion
            "written_sections": {},
            "current_section_data": {},
            "retrieved_info": {},
            "history": []
        }
        
        # Define the ReAct loop
        max_iterations = 50  # Safety limit
        current_iteration = 0
        
        # Start the ReAct loop
        print(f"======= Starting Writer Agent =======")
        print(f"Tutorial: {tutorial_plan.get('title', 'Untitled')}")
        print(f"Sections to write: {state['sections_to_write']}")
        
        while current_iteration < max_iterations:
            current_iteration += 1
            print(f"\n----- Writer Iteration {current_iteration} -----")
            
            # 1. Observe: Analyze the current state
            observation = self._observe(state)
            print(f"Observation: {observation}")
            
            # 2. Think: Reason about what to write next
            thought, next_action, action_input = self._think(state, observation)
            print(f"Thought: {thought}")
            print(f"Next Action: {next_action}")
            
            # Check if we're done
            if next_action == "finish":
                print(f"Writer agent has completed all sections.")
                break
            
            # 3. Act: Execute a writing action
            result = self._act(next_action, action_input, state)
            
            # 4. Update state with the result
            self._update_state(state, next_action, result)
            
            # Add to history
            state["history"].append({
                "iteration": current_iteration,
                "observation": observation,
                "thought": thought,
                "action": next_action,
                "result": "Success" if result and "error" not in result else "Failed"
            })
        
        # Return all written sections
        final_result = {
            "title": tutorial_plan.get("title", "Tutorial"),
            "sections": state["written_sections"]
        }
        
        # Store the last result for potential retrieval by the sequential agent
        self.last_result = final_result
        
        return final_result
    
    def _observe(self, state: Dict[str, Any]) -> str:
        """
        Observe the current state and return a description.
        
        Args:
            state: Current state of the agent
            
        Returns:
            String description of the current state
        """
        current_step = state["current_step"]
        section_index = state["section_index"]
        sections_written = len([s for s in state["written_sections"] if s != "error"])
        sections_to_write = state["sections_to_write"]
        
        if current_step == "initialize":
            return f"Writer agent initialized. Ready to start writing the tutorial. {sections_to_write} sections to write."
        
        if current_step == "retrieve_information":
            section_name = self._get_section_name(state)
            return f"Retrieved information for {section_name}. Ready to write content."
        
        if current_step == "write_section":
            section_name = self._get_section_name(state)
            return f"Completed writing {section_name}. Progress: {sections_written}/{sections_to_write} sections."
        
        return f"Current step: {current_step}. Section index: {section_index}. Progress: {sections_written}/{sections_to_write} sections."
    
    def _think(self, state: Dict[str, Any], observation: str) -> tuple:
        """
        Think about the next action based on the current state and observation.
        
        Args:
            state: Current state of the agent
            observation: Current observation
            
        Returns:
            Tuple of (thought, next_action, action_input)
        """
        from langchain.schema import HumanMessage, SystemMessage
        
        tutorial_plan = state["tutorial_plan"]
        current_step = state["current_step"]
        section_index = state["section_index"]
        sections_written = len([s for s in state["written_sections"] if s != "error"])
        sections_to_write = state["sections_to_write"]
        
        # Determine the next logical action based on the state
        if current_step == "initialize" or current_step == "write_section":
            # After initializing or finishing a section, move to the next section
            new_section_index = section_index + 1
            
            # Check if we've written all sections
            if new_section_index >= sections_to_write - 1:  # -1 because conclusion is handled separately
                if "conclusion" not in state["written_sections"]:
                    thought = "All main sections are complete. Now I need to write the conclusion."
                    return thought, "write_conclusion", tutorial_plan
                else:
                    thought = "All sections including introduction and conclusion are complete. The tutorial is finished."
                    return thought, "finish", None
                    
            # Determine what section to write next
            if new_section_index == 0:  # After intro, first regular section
                thought = "Introduction is complete. Now I need to start writing the first section."
                section_data = tutorial_plan["sections"][0]
                state["section_index"] = new_section_index
                state["current_section_data"] = section_data
                return thought, "retrieve_information", section_data
            else:
                thought = f"Section {new_section_index} is next. I need to gather information for it."
                section_data = tutorial_plan["sections"][new_section_index]
                state["section_index"] = new_section_index
                state["current_section_data"] = section_data
                return thought, "retrieve_information", section_data
        
        elif current_step == "retrieve_information":
            # After retrieving information, write the section
            thought = f"I have gathered relevant information. Now I can write the section content."
            return thought, "write_section", {
                "section_data": state["current_section_data"],
                "retrieved_info": state["retrieved_info"]
            }
            
        elif section_index == -1:
            # Special case for the introduction
            thought = "Need to start by writing the introduction for the tutorial."
            return thought, "write_introduction", tutorial_plan
        
        # Default case (should rarely reach here)
        thought = "Analyzing the current state to determine the next action."
        prompt = f"""
        You are a writer agent working on a tutorial.
        
        Current state:
        - Progress: {sections_written}/{sections_to_write} sections written
        - Current section index: {section_index}
        - Last step: {current_step}
        
        What should I do next?
        """
        
        response = self.llm.invoke([
            SystemMessage(content="You are a helpful assistant. Analyze the current state and recommend the next action."),
            HumanMessage(content=prompt)
        ])
        
        # Default to retrieving information for the current section if unsure
        if section_index >= 0 and section_index < len(tutorial_plan["sections"]):
            section_data = tutorial_plan["sections"][section_index]
            return response.content, "retrieve_information", section_data
        else:
            return response.content, "write_introduction", tutorial_plan
    
    def _act(self, action: str, action_input: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a writing action.
        
        Args:
            action: Name of the action to execute
            action_input: Input for the action
            state: Current state of the agent
            
        Returns:
            Result of the action
        """
        try:
            if action == "retrieve_information":
                section_data = action_input
                return self.writer_tool.retrieve_information(
                    section_title=section_data.get("title", ""),
                    section_summary=section_data.get("content_summary", ""),
                    key_points=section_data.get("key_points", [])
                )
                
            elif action == "write_section":
                section_data = action_input["section_data"]
                retrieved_info = action_input["retrieved_info"]
                return self.writer_tool.write_section(
                    section_data=section_data,
                    retrieved_info=retrieved_info
                )
                
            elif action == "write_introduction":
                tutorial_plan = action_input
                return self.writer_tool.write_introduction(tutorial_plan)
                
            elif action == "write_conclusion":
                tutorial_plan = action_input
                return self.writer_tool.write_conclusion(tutorial_plan)
                
            return {"error": f"Unknown action: {action}"}
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error executing {action}: {e}")
            logger.error(error_trace)
            return {"error": str(e), "traceback": error_trace}
    
    def _update_state(self, state: Dict[str, Any], action: str, result: Dict[str, Any]) -> None:
        """
        Update the state based on the action result.
        
        Args:
            state: Current state of the agent
            action: Action that was executed
            result: Result of the action
        """
        state["current_step"] = action
        
        if action == "retrieve_information":
            state["retrieved_info"] = result
            
        elif action == "write_section":
            section_index = state["section_index"]
            section_key = f"section_{section_index}"
            state["written_sections"][section_key] = result
            
        elif action == "write_introduction":
            state["written_sections"]["introduction"] = result
            state["section_index"] = -1  # Introduction is complete
            
        elif action == "write_conclusion":
            state["written_sections"]["conclusion"] = result
            
    def _get_section_name(self, state: Dict[str, Any]) -> str:
        """
        Get the name of the current section being worked on.
        
        Args:
            state: Current state of the agent
            
        Returns:
            Name of the current section
        """
        section_index = state["section_index"]
        
        if section_index == -1:
            return "Introduction"
            
        if section_index >= len(state["tutorial_plan"]["sections"]):
            return "Conclusion"
            
        if section_index >= 0:
            return state["tutorial_plan"]["sections"][section_index].get("title", f"Section {section_index+1}")
            
        return f"Section {section_index+1}"
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

import os
import json
import re

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

class SequentialReActAgent:
    """
    A sequential agent that follows the ReAct pattern:
    1. Observe: Analyze the current state
    2. Think: Reason about what to do next
    3. Act: Execute a tool
    4. Repeat until the goal is reached
    """
    
    def __init__(
        self,
        llm_model: str = "llama3.2:latest",
        collection_name: str = "tutorial_collection",
        text_embedding_model: str = "nomic-embed-text",
        image_embedding_model: str = "llava"
    ):
        """
        Initialize the sequential ReAct agent.
        
        Args:
            llm_model: Name of the LLM model
            collection_name: Name of the ChromaDB collection
            text_embedding_model: Name of the text embedding model
            image_embedding_model: Name of the image embedding model
        """
        self.llm_model = llm_model
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline(
            collection_name=collection_name,
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            llm_model=llm_model
        )
        
        # Initialize tools
        self.tools = {
            "extract_pdf": SimplePDFExtractor(),
            "summarize_pdf": SimplePDFSummarizer(llm_model),
            "plan_tutorial": SimpleTutorialPlanner(llm_model),
            "write_tutorial": SimpleTutorialWriter(llm_model, self.rag_pipeline),
            "format_tutorial": SimpleTutorialFormatter(llm_model),
            "improve_tutorial": SimpleTutorialImprover(llm_model)
        }
        
        # Import LLM for reasoning
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(model=self.llm_model)
    
    def run(self, pdf_path: str, goal: str) -> Dict[str, Any]:
        """
        Run the sequential ReAct agent to generate a tutorial.
        
        Args:
            pdf_path: Path to the PDF file
            goal: Description of the tutorial goal
            
        Returns:
            Dictionary with the generated tutorial and metadata
        """
        # Initialize the state
        state = {
            "pdf_path": pdf_path,
            "goal": goal,
            "current_step": "initialize",
            "history": [],
            "tools_results": {},
            "final_tutorial": "",
            "tutorial_plan": {},
            "pdf_content": None,
            "pdf_summary": None
        }
        
        # Define the ReAct loop
        max_iterations = 12  # Safety limit to prevent infinite loops
        current_iteration = 0
        
        # Start the ReAct loop
        print(f"======= Starting ReAct Agent for Tutorial Generation =======")
        print(f"Goal: {goal}")
        print(f"PDF Path: {pdf_path}")
        
        while current_iteration < max_iterations:
            current_iteration += 1
            print(f"\n----- Iteration {current_iteration} -----")
            
            # 1. Observe: Analyze the current state
            observation = self._observe(state)
            print(f"Observation: {observation}")
            
            # 2. Think: Reason about what to do next
            thought, next_action, action_input = self._think(state, observation)
            print(f"Thought: {thought}")
            print(f"Next Action: {next_action}")
            
            # Check if we're done
            if next_action == "finish":
                print(f"Agent determined the task is complete.")
                break
            
            # 3. Act: Execute a tool
            if next_action in self.tools:
                print(f"Executing tool: {next_action}")
                result = self._act(next_action, action_input, state)
                
                # Update state with the result
                state["tools_results"][next_action] = result
                state["current_step"] = next_action
                state["history"].append({
                    "iteration": current_iteration,
                    "observation": observation,
                    "thought": thought,
                    "action": next_action,
                    "result": "Success" if result else "Failed"
                })
                
                # Special state updates based on the tool
                if next_action == "extract_pdf":
                    state["pdf_content"] = result
                elif next_action == "summarize_pdf":
                    state["pdf_summary"] = result
                elif next_action == "plan_tutorial":
                    state["tutorial_plan"] = result
                elif next_action == "improve_tutorial":
                    state["final_tutorial"] = result.get("improved_content", "")
            else:
                print(f"Unknown action: {next_action}")
                state["history"].append({
                    "iteration": current_iteration,
                    "observation": observation,
                    "thought": thought,
                    "action": next_action,
                    "result": "Failed - Unknown action"
                })
        
        # Extract the final tutorial and other relevant information
        final_result = {
            "tutorial": state.get("final_tutorial", ""),
            "improvements": state.get("tools_results", {}).get("improve_tutorial", {}).get("improvements", []),
            "plan": state.get("tutorial_plan", {}),
            "extracted_knowledge": state.get("pdf_summary", {})
        }
        
        # If we reached max iterations without finishing, add a note
        if current_iteration >= max_iterations:
            final_result["warning"] = "Maximum iterations reached without completing the task."
        
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
        history = state["history"]
        
        if current_step == "initialize":
            return f"Agent initialized. Ready to process the PDF at {state['pdf_path']}. The goal is to create a tutorial: {state['goal']}"
        
        # Check tool results
        if current_step in state["tools_results"]:
            result = state["tools_results"][current_step]
            
            if current_step == "extract_pdf":
                return f"PDF extraction complete. Extracted {len(result.get('pages', []))} pages and {len(result.get('images', []))} images."
            
            elif current_step == "summarize_pdf":
                return f"PDF summarization complete. Main topics: {', '.join(result.get('overall_summary', {}).get('main_topics', ['None'])[:3])}"
            
            elif current_step == "plan_tutorial":
                sections = result.get("sections", [])
                return f"Tutorial planning complete. Created a plan with {len(sections)} sections: {', '.join([s.get('title', '') for s in sections[:3]])}"
            
            elif current_step == "write_tutorial":
                sections = result.get("sections", {})
                return f"Tutorial writing complete. Wrote {len(sections)} sections including introduction and conclusion."
            
            elif current_step == "format_tutorial":
                return "Tutorial formatting complete. Added proper markdown formatting, code blocks, and structure."
            
            elif current_step == "improve_tutorial":
                improvements = result.get("improvements", [])
                return f"Tutorial improvement complete. Made {len(improvements)} improvements to enhance quality."
        
        # If we reach here, we don't have specific information
        completed_steps = [h["action"] for h in history]
        return f"Current step: {current_step}. Completed steps: {', '.join(completed_steps)}"
    
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
        
        # Create a prompt for the thinking process
        prompt = f"""
        You are a tutorial generation agent working with a PDF document.
        
        Goal: {state["goal"]}
        
        Current observation: {observation}
        
        Current step: {state["current_step"]}
        
        Available tools:
        - extract_pdf: Extract content from a PDF file
        - summarize_pdf: Summarize the extracted PDF content
        - plan_tutorial: Plan the structure of a tutorial based on PDF content
        - write_tutorial: Write individual sections of a tutorial based on the plan
        - format_tutorial: Format the tutorial with proper markdown and structure
        - improve_tutorial: Identify and implement improvements to the tutorial
        - finish: Complete the tutorial generation process
        
        History: {json.dumps([h for h in state["history"]], indent=2)}
        
        Based on the current state and observation, reason step by step about what to do next.
        Your output should be in the following format:
        
        Thought: <your detailed reasoning process>
        Action: <the tool to use or "finish" if done>
        ActionInput: <input parameters for the tool as JSON or "None" if finishing>
        """
        
        response = self.llm.invoke([
            SystemMessage(content="""
            You are an expert tutorial generation agent. You decide the next step in the tutorial generation process.
            Think carefully about what information you have and what information you still need.
            The typical flow is: extract_pdf → summarize_pdf → plan_tutorial → write_tutorial → format_tutorial → improve_tutorial → finish.
            Only select finish when all other steps have been completed.
            
            Your output must strictly follow this format:
            Thought: <your detailed reasoning>
            Action: <tool name>
            ActionInput: <parameters as JSON or "None">
            """),
            HumanMessage(content=prompt)
        ])
        
        # Parse the response
        content = response.content
        
        # Extract thought, action, and action_input
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', content, re.DOTALL)
        action_match = re.search(r'Action:\s*(.*?)(?=ActionInput:|$)', content, re.DOTALL)
        action_input_match = re.search(r'ActionInput:\s*(.*?)(?=$)', content, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        action_input_str = action_input_match.group(1).strip() if action_input_match else ""
        
        # Process action input
        action_input = None
        if action_input_str and action_input_str != "None":
            try:
                action_input = json.loads(action_input_str)
            except:
                # If JSON parsing fails, use the string as is
                action_input = action_input_str
        
        # Prepare the action input based on the specific action
        if action == "extract_pdf":
            action_input = state["pdf_path"]
        elif action == "summarize_pdf":
            action_input = state["pdf_content"]
        elif action == "plan_tutorial":
            action_input = {
                "summarized_content": state["pdf_summary"],
                "tutorial_goal": state["goal"]
            }
        elif action == "write_tutorial":
            action_input = {
                "tutorial_plan": state["tutorial_plan"],
                "pdf_content": state["pdf_content"]
            }
        elif action == "format_tutorial":
            action_input = {
                "written_sections": state["tools_results"]["write_tutorial"],
                "pdf_content": state["pdf_content"]
            }
        elif action == "improve_tutorial":
            action_input = state["tools_results"]["format_tutorial"]
        
        return thought, action, action_input
    
    def _act(self, action: str, action_input: Any, state: Dict[str, Any]) -> Any:
        """
        Execute a tool based on the selected action.
        
        Args:
            action: Name of the tool to execute
            action_input: Input for the tool
            state: Current state of the agent
            
        Returns:
            Result of the tool execution
        """
        try:
            tool = self.tools[action]
            
            # Call the appropriate method based on the action
            if action == "extract_pdf":
                return tool.extract(action_input)
            elif action == "summarize_pdf":
                return tool.summarize(action_input)
            elif action == "plan_tutorial":
                return tool.plan(**action_input)
            elif action == "write_tutorial":
                return tool.write(**action_input)
            elif action == "format_tutorial":
                return tool.format(**action_input)
            elif action == "improve_tutorial":
                return tool.improve(action_input)
            else:
                return {"error": f"Unrecognized action: {action}"}
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error executing {action}: {e}")
            print(error_trace)
            return {"error": str(e), "traceback": error_trace}
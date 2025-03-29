"""
Planning agent for tutorial generation.
"""
import logging
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PlanningAgent:
    """Agent responsible for planning the tutorial structure."""
    
    def __init__(self, model_provider, tools):
        """Initialize planning agent.
        
        Args:
            model_provider: Provider for accessing models
            tools: Tool registry for accessing tools
        """
        self.model_provider = model_provider
        self.tools = tools
        self.llm = model_provider.get_llm()
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning phase - create tutorial outline.
        
        Args:
            state (Dict[str, Any]): Current agent state
            
        Returns:
            Dict[str, Any]: Updated agent state
        """
        logger.info("Planning agent executing")
        
        # ReAct loop components
        thoughts = []
        actions = []
        observations = []
        
        # Extract query and PDF content from state
        query = state["query"]
        pdf_content = state["pdf_content"]
        
        # ===== THOUGHT =====
        initial_thought = self._think_about_tutorial(query, pdf_content)
        thoughts.append(initial_thought)
        
        # ===== ACTION =====
        # Analyze PDF structure to inform the plan
        action = {
            "tool": "analyze_pdf_structure",
            "input": pdf_content["text"]
        }
        actions.append(action)
        
        # ===== OBSERVATION =====
        # Execute the action and observe results
        pdf_analysis = self.tools.execute_tool(action["tool"], action["input"])
        observations.append(pdf_analysis)
        
        # ===== THOUGHT =====
        # Consider document structure and content for the plan
        structure_thought = self._think_about_structure(query, pdf_analysis)
        thoughts.append(structure_thought)
        
        # ===== ACTION =====
        # Extract any table of contents if available
        toc_action = {
            "tool": "extract_document_toc",
            "input": pdf_content.get("toc", [])
        }
        actions.append(toc_action)
        
        # ===== OBSERVATION =====
        toc_result = pdf_content.get("toc", [])
        observations.append(toc_result)
        
        # ===== FINAL THOUGHT =====
        # Create final tutorial plan
        final_thought = self._create_tutorial_plan(query, pdf_analysis, toc_result, initial_thought)
        thoughts.append(final_thought)
        
        # Generate tutorial plan
        plan_sections = self._extract_plan_sections(final_thought)
        
        # Update state with plan
        new_state = state.copy()
        new_state["plan"] = plan_sections
        new_state["current_section"] = plan_sections[0] if plan_sections else ""
        new_state["tutorial_content"] = {}
        new_state["observations"] = state.get("observations", []) + [{
            "agent": "planning",
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "result": plan_sections
        }]
        
        return new_state
    
    def _think_about_tutorial(self, query: str, pdf_content: Dict[str, Any]) -> str:
        """Think about what the tutorial should include.
        
        Args:
            query (str): User query
            pdf_content (Dict[str, Any]): Processed PDF content
            
        Returns:
            str: Thought about tutorial requirements
        """
        prompt = f"""
        You are a planning agent tasked with creating a tutorial outline.
        The tutorial topic is: {query}
        
        The tutorial will be based on information from a PDF document with the following metadata:
        - Title: {pdf_content.get("metadata", {}).get("title", "Unknown")}
        - Pages: {pdf_content.get("pages", 0)}
        - Contains {len(pdf_content.get("images", []))} images and {len(pdf_content.get("tables", []))} tables
        
        Think about what should be included in this tutorial:
        1. What is the main focus of the tutorial?
        2. Who is the target audience?
        3. What prerequisite knowledge might be needed?
        4. What skills or knowledge should the reader gain?
        5. What components or sections would make for a logical progression?
        
        Provide your thoughts on these questions.
        """
        
        try:
            thought = str(self.llm.complete(prompt))
            return thought
        except Exception as e:
            logger.error(f"Error in _think_about_tutorial: {e}")
            return f"I need to create a tutorial about {query}. I'll analyze the PDF content to determine the best structure."
    
    def _think_about_structure(self, query: str, pdf_analysis: Dict[str, Any]) -> str:
        """Think about the document structure and how to organize the tutorial.
        
        Args:
            query (str): User query
            pdf_analysis (Dict[str, Any]): Analysis of PDF structure
            
        Returns:
            str: Thought about document structure
        """
        headings_text = "\n".join([f"- {h.get('text', '')}" for h in pdf_analysis.get("headings", [][:10])])
        
        prompt = f"""
        I'm planning a tutorial on: {query}
        
        The PDF document has the following potential section headings:
        {headings_text}
        
        Based on these headings and the tutorial topic, how should I structure the tutorial to ensure:
        1. A clear, logical progression of concepts
        2. Appropriate coverage of the important topics
        3. A good balance between theory and practical examples
        
        Provide your thoughts on organizing the tutorial structure.
        """
        
        try:
            thought = str(self.llm.complete(prompt))
            return thought
        except Exception as e:
            logger.error(f"Error in _think_about_structure: {e}")
            return f"Looking at the document headings, I can see several key topics that should be included in the tutorial. I'll organize them in a logical sequence."
    
    def _create_tutorial_plan(self, 
                             query: str, 
                             pdf_analysis: Dict[str, Any],
                             toc: List[Dict[str, Any]],
                             initial_thought: str) -> str:
        """Create the final tutorial plan.
        
        Args:
            query (str): User query
            pdf_analysis (Dict[str, Any]): Analysis of PDF structure
            toc (List[Dict[str, Any]]): Table of contents
            initial_thought (str): Initial thought about the tutorial
            
        Returns:
            str: Final thought with tutorial plan
        """
        # Format TOC if available
        toc_text = ""
        if toc:
            toc_items = []
            for item in toc[:10]:  # Limit to first 10 for brevity
                level = item.get("level", 1)
                indent = "  " * (level - 1)
                toc_items.append(f"{indent}- {item.get('title', '')}")
            toc_text = "\n".join(toc_items)
        
        prompt = f"""
        I need to create a detailed plan for a tutorial on: {query}
        
        Based on my analysis and the document structure, here is what I know:
        
        {initial_thought}
        
        Document Table of Contents:
        {toc_text}
        
        Create a tutorial plan with 5-8 clear sections. For each section:
        1. Provide a descriptive title
        2. Add a brief description of what will be covered
        3. Note any images or tables that should be included
        
        Format your response as a JSON list of sections:
        [
          {{
            "title": "Section Title",
            "description": "Brief description of what this section covers",
            "includes": ["concepts", "examples", "images", "tables"]
          }},
          ...
        ]
        """
        
        try:
            thought = str(self.llm.complete(prompt))
            return thought
        except Exception as e:
            logger.error(f"Error in _create_tutorial_plan: {e}")
            return """
            [
              {
                "title": "Introduction",
                "description": "Overview of the tutorial and what will be covered",
                "includes": ["concepts"]
              },
              {
                "title": "Getting Started",
                "description": "Basic setup and prerequisites",
                "includes": ["examples"]
              },
              {
                "title": "Core Concepts",
                "description": "Key concepts and theory",
                "includes": ["concepts", "examples", "images"]
              },
              {
                "title": "Practical Application",
                "description": "Hands-on implementation",
                "includes": ["examples", "tables"]
              },
              {
                "title": "Advanced Techniques",
                "description": "More sophisticated approaches",
                "includes": ["concepts", "examples"]
              },
              {
                "title": "Conclusion",
                "description": "Summary and next steps",
                "includes": ["concepts"]
              }
            ]
            """
    
    def _extract_plan_sections(self, plan_text: str) -> List[str]:
        """Extract section titles from plan JSON.
        
        Args:
            plan_text (str): Text containing plan JSON
            
        Returns:
            List[str]: List of section titles
        """
        # Try to find JSON in the text
        try:
            # Look for JSON array in the text
            start_idx = plan_text.find("[")
            end_idx = plan_text.rfind("]") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = plan_text[start_idx:end_idx]
                sections_data = json.loads(json_str)
                
                # Extract titles and descriptions
                sections = []
                for section in sections_data:
                    title = section.get("title", "")
                    description = section.get("description", "")
                    sections.append(f"{title}: {description}")
                
                return sections
            
        except Exception as e:
            logger.warning(f"Error extracting JSON plan: {e}")
        
        # Fallback to simple parsing if JSON extraction fails
        try:
            # Look for section patterns in text
            import re
            section_pattern = re.compile(r"(\d+\.\s+|##\s+|Section\s+\d+:\s+)([^\n]+)")
            matches = section_pattern.findall(plan_text)
            
            if matches:
                return [match[1].strip() for match in matches]
            
            # Try to find lines that might be section titles
            lines = plan_text.split('\n')
            sections = []
            for line in lines:
                line = line.strip()
                if line and len(line) < 100 and not line.startswith(("I ", "The ", "This ", "In ", "Based ", "From ")):
                    sections.append(line)
            
            return sections[:8]  # Limit to 8 sections
            
        except Exception as e:
            logger.warning(f"Error extracting sections with regex: {e}")
        
        # Final fallback
        return [
            "Introduction: Overview of the topic",
            "Background: Essential concepts and theory",
            "Main Components: Key elements and their functions",
            "Implementation: Step-by-step guide",
            "Examples: Practical applications",
            "Advanced Topics: Further exploration",
            "Conclusion: Summary and next steps"
        ]
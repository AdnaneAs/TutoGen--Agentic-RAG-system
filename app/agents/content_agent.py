"""
Content generation agent for creating tutorial content.
"""
import logging
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ContentAgent:
    """Agent responsible for generating tutorial content."""
    
    def __init__(self, model_provider, tools, rag_pipeline):
        """Initialize content generation agent.
        
        Args:
            model_provider: Provider for accessing models
            tools: Tool registry for accessing tools
            rag_pipeline: RAG pipeline for content retrieval
        """
        self.model_provider = model_provider
        self.tools = tools
        self.rag_pipeline = rag_pipeline
        self.llm = model_provider.get_llm()
        self.config = None  # Will be set from state
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content generation for current section.
        
        Args:
            state (Dict[str, Any]): Current agent state
            
        Returns:
            Dict[str, Any]: Updated agent state
        """
        logger.info(f"Content agent executing for section: {state['current_section']}")
        
        # Store config for later use
        self.config = state.get("config")
        
        # ReAct loop components
        thoughts = []
        actions = []
        observations = []
        
        # Extract information from state
        current_section = state["current_section"]
        section_title = current_section.split(":", 1)[0].strip()
        research_results = state.get("research", {})
        research_summary = research_results.get("content", "")
        
        # ===== THOUGHT =====
        # Think about how to structure this section
        structure_thought = self._think_about_section_structure(current_section, research_summary)
        thoughts.append(structure_thought)
        
        # ===== ACTION =====
        # Prepare any visual elements for inclusion
        visuals_action = {
            "tool": "prepare_visual_elements",
            "input": {
                "visual_elements": research_results.get("visual_elements", []),
                "section": current_section
            }
        }
        actions.append(visuals_action)
        
        # ===== OBSERVATION =====
        # Process visual elements and get markdown references
        visual_elements = research_results.get("visual_elements", [])
        markdown_references = self._prepare_visual_references(visual_elements)
        observations.append(markdown_references)
        
        # ===== THOUGHT =====
        # Plan the content with visual elements integrated
        content_plan = self._plan_section_content(
            current_section, 
            research_summary,
            markdown_references
        )
        thoughts.append(content_plan)
        
        # ===== ACTION =====
        # Generate the section content
        generation_action = {
            "tool": "generate_section_content",
            "input": {
                "section": current_section,
                "research": research_summary,
                "visual_references": markdown_references,
                "content_plan": content_plan
            }
        }
        actions.append(generation_action)
        
        # ===== OBSERVATION =====
        # Generate the content
        section_content = self._generate_section_content(
            current_section,
            research_summary,
            markdown_references,
            content_plan
        )
        observations.append(section_content)
        
        # ===== FINAL ACTION =====
        # Update the tutorial document
        if "tutorial_document" not in state:
            # Create new document if not exists
            tutorial_title = f"Tutorial: {state['query']}"
            tutorial_document = self._create_tutorial_document(tutorial_title)
        else:
            tutorial_document = state["tutorial_document"]
        
        document_action = {
            "tool": "update_tutorial_document",
            "input": {
                "document": tutorial_document,
                "section_title": section_title,
                "content": section_content
            }
        }
        actions.append(document_action)
        
        # ===== FINAL OBSERVATION =====
        # Update the document
        updated_document = self._update_tutorial_document(
            tutorial_document,
            section_title,
            section_content
        )
        observations.append(updated_document)
        
        # Move to next section or complete
        plan = state["plan"]
        current_index = plan.index(current_section) if current_section in plan else -1
        
        new_state = state.copy()
        new_state["tutorial_document"] = updated_document
        
        # Update tutorial content dictionary
        if "tutorial_content" not in new_state:
            new_state["tutorial_content"] = {}
        new_state["tutorial_content"][section_title] = section_content
        
        # Check if there are more sections
        if current_index < len(plan) - 1:
            new_state["current_section"] = plan[current_index + 1]
        else:
            # No more sections, we're done
            new_state["completed"] = True
            # Finalize document
            markdown_generator = self.tools.get_tool("markdown_generator")
            if markdown_generator:
                markdown_generator.finalize_document(updated_document)
        
        # Add observations to state
        new_state["observations"] = state.get("observations", []) + [{
            "agent": "content",
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "section": section_title,
            "content": section_content
        }]
        
        return new_state
    
    def _think_about_section_structure(self, section: str, research: str) -> str:
        """Think about how to structure this section.
        
        Args:
            section (str): Current section title and description
            research (str): Research summary
            
        Returns:
            str: Thought about section structure
        """
        prompt = f"""
        I'm creating content for this tutorial section:
        {section}
        
        Based on my research:
        {research[:500]}...
        
        I need to think about:
        1. What is the most logical structure for this section?
        2. What key points need to be covered?
        3. How should I balance explanations, examples, and visual elements?
        4. What would make this section most valuable to the reader?
        
        Provide your thoughts on structuring this section effectively.
        """
        
        try:
            thought = str(self.llm.complete(prompt))
            return thought
        except Exception as e:
            logger.error(f"Error thinking about section structure: {e}")
            return f"For the section '{section.split(':')[0]}', I should structure it with a clear introduction, followed by key concepts, practical examples, and a summary of important points. I'll integrate visual elements where they help illustrate the concepts."
    
    def _prepare_visual_references(self, visual_elements: List[Dict[str, Any]]) -> Dict[str, str]:
        """Prepare markdown references for visual elements.
        
        Args:
            visual_elements (List[Dict[str, Any]]): Visual elements from research
            
        Returns:
            Dict[str, str]: Markdown references for visual elements
        """
        markdown_references = {}
        
        try:
            markdown_generator = self.tools.get_tool("markdown_generator")
            if not markdown_generator:
                return markdown_references
            
            # Process each visual element
            for i, element in enumerate(visual_elements):
                element_type = element.get("type")
                
                if element_type == "image" and "path" in element:
                    # Add image
                    image_path = element["path"]
                    if os.path.exists(image_path):
                        alt_text = f"Image for {element.get('caption', 'tutorial')}"
                        markdown_code = markdown_generator.add_image(
                            document={"assets_dir": os.path.dirname(image_path)},
                            image_path=image_path,
                            alt_text=alt_text,
                            caption=element.get("caption")
                        )
                        ref_id = f"image_{i}"
                        markdown_references[ref_id] = markdown_code
                
                elif element_type == "table" and "markdown" in element:
                    # Add table
                    table_markdown = element["markdown"]
                    caption = element.get("caption", f"Table {i+1}")
                    markdown_code = markdown_generator.add_table(
                        document={},
                        table_data=table_markdown,
                        caption=caption
                    )
                    ref_id = f"table_{i}"
                    markdown_references[ref_id] = markdown_code
        
        except Exception as e:
            logger.error(f"Error preparing visual references: {e}")
        
        return markdown_references
    
    def _plan_section_content(self, 
                             section: str, 
                             research: str,
                             visual_references: Dict[str, str]) -> str:
        """Plan the section content structure.
        
        Args:
            section (str): Current section title and description
            research (str): Research summary
            visual_references (Dict[str, str]): Markdown references for visual elements
            
        Returns:
            str: Plan for section content
        """
        # Create a list of available visual elements
        visuals_list = ""
        for ref_id, markdown in visual_references.items():
            visuals_list += f"- {ref_id}: {markdown[:50]}...\n"
        
        prompt = f"""
        I'm creating content for this tutorial section:
        {section}
        
        Based on my research findings:
        {research[:800]}...
        
        Available visual elements:
        {visuals_list}
        
        Create a detailed plan for this section that includes:
        1. Introduction to the section topics
        2. Key concepts to explain
        3. Examples or applications to include
        4. Where to integrate the visual elements
        5. Conclusion or summary points
        
        Provide a structured outline for the section content.
        """
        
        try:
            plan = str(self.llm.complete(prompt))
            return plan
        except Exception as e:
            logger.error(f"Error planning section content: {e}")
            return f"""
            Section Plan: {section.split(':')[0]}
            
            1. Introduction
               - Brief overview of the topic
               - Why it's important
               
            2. Key Concepts
               - Main ideas and definitions
               - Theoretical background
               
            3. Examples and Applications
               - Practical use cases
               - Code samples if applicable
               
            4. Visual Illustrations
               - Include relevant images and tables
               
            5. Summary
               - Key takeaways
               - Connection to next section
            """
    
    def _generate_section_content(self,
                                section: str,
                                research: str,
                                visual_references: Dict[str, str],
                                content_plan: str) -> str:
        """Generate the section content.
        
        Args:
            section (str): Current section title and description
            research (str): Research summary
            visual_references (Dict[str, str]): Markdown references for visual elements
            content_plan (str): Content plan
            
        Returns:
            str: Generated section content
        """
        # Extract section title
        section_title = section.split(":", 1)[0].strip()
        
        # Create a reference guide for visual elements
        visuals_guide = ""
        for ref_id, markdown in visual_references.items():
            # Only include a preview of the markdown
            preview = markdown.split("\n")[0] if "\n" in markdown else markdown[:50]
            visuals_guide += f"- {ref_id}: {preview}...\n"
        
        prompt = f"""
        Generate comprehensive content for this tutorial section:
        # {section_title}
        
        Base your content on:
        
        Research findings:
        {research}
        
        Content plan:
        {content_plan}
        
        Available visual elements (reference them using their IDs):
        {visuals_guide}
        
        Create the complete section content in markdown format. Be detailed, informative, and educational.
        Include appropriate headings (using ## for subsections, ### for sub-subsections).
        Integrate visual elements at appropriate points using their reference IDs in ALL CAPS (e.g., IMAGE_0, TABLE_1).
        Don't include the section title heading, as it will be added automatically.
        
        Generate well-structured, educational tutorial content that balances explanations, examples, and visual elements.
        """
        
        try:
            # Generate content
            content = str(self.llm.complete(prompt))
            
            # Replace visual element placeholders with actual markdown
            for ref_id, markdown in visual_references.items():
                placeholder = ref_id.upper()
                content = content.replace(placeholder, markdown)
            
            return content
        except Exception as e:
            logger.error(f"Error generating section content: {e}")
            return f"""
            This section covers {section_title}, an important topic for understanding the overall subject.
            
            ## Key Concepts
            
            The key concepts involve understanding the fundamental principles and how they relate to the broader topic.
            
            ## Practical Applications
            
            These concepts can be applied in various scenarios to solve real-world problems.
            
            ## Summary
            
            This section provided an overview of {section_title} and its importance in the context of the tutorial.
            """
    
    def _create_tutorial_document(self, title: str) -> Dict[str, Any]:
        """Create a new tutorial document.
        
        Args:
            title (str): Document title
            
        Returns:
            Dict[str, Any]: Document metadata
        """
        try:
            markdown_generator = self.tools.get_tool("markdown_generator")
            if markdown_generator:
                document = markdown_generator.create_document(title)
                return document
            else:
                # Fallback if tool not available
                return {
                    "title": title,
                    "path": f"tutorial_{title.lower().replace(' ', '_')}.md",
                    "sections": {}
                }
        except Exception as e:
            logger.error(f"Error creating tutorial document: {e}")
            return {
                "title": title,
                "path": f"tutorial_{title.lower().replace(' ', '_')}.md",
                "sections": {}
            }
    
    def _update_tutorial_document(self, 
                                document: Dict[str, Any], 
                                section_title: str, 
                                content: str) -> Dict[str, Any]:
        """Update the tutorial document with section content.
        
        Args:
            document (Dict[str, Any]): Document metadata
            section_title (str): Section title
            content (str): Section content
            
        Returns:
            Dict[str, Any]: Updated document metadata
        """
        try:
            markdown_generator = self.tools.get_tool("markdown_generator")
            if markdown_generator:
                # Check if section exists
                if section_title in document.get("sections", {}):
                    updated_doc = markdown_generator.update_section(document, section_title, content)
                else:
                    updated_doc = markdown_generator.add_section(document, section_title, content)
                return updated_doc
            else:
                # Fallback if tool not available
                doc_path = document.get("path", "tutorial.md")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(doc_path), exist_ok=True)
                
                # Read existing content if file exists
                if os.path.exists(doc_path):
                    with open(doc_path, "r", encoding="utf-8") as f:
                        doc_content = f.read()
                else:
                    doc_content = f"# {document.get('title', 'Tutorial')}\n\n"
                
                # Add or update section
                section_header = f"## {section_title}"
                if section_header in doc_content:
                    # Replace existing section
                    import re
                    pattern = re.compile(f"{re.escape(section_header)}.*?(?=\n## |$)", re.DOTALL)
                    doc_content = pattern.sub(f"{section_header}\n\n{content}", doc_content)
                else:
                    # Add new section
                    doc_content += f"\n\n{section_header}\n\n{content}"
                
                # Write updated content
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(doc_content)
                
                # Update document metadata
                document.setdefault("sections", {})[section_title] = {
                    "title": section_title,
                    "id": section_title.lower().replace(" ", "-")
                }
                
                return document
                
        except Exception as e:
            logger.error(f"Error updating tutorial document: {e}")
            return document
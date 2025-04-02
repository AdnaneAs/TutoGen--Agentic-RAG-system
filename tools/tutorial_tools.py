from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import os
import json
import re

# Import RAG pipeline (using relative imports)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.rag_pipeline import RAGPipeline


class SimpleTutorialPlanner:
    """Simple tool to plan the structure of a tutorial based on PDF content."""
    
    def __init__(self, llm_model: str):
        """
        Initialize tutorial planner.
        
        Args:
            llm_model: Name of the LLM model to use for planning
        """
        self.llm_model = llm_model
    
    def plan(self, summarized_content: Dict[str, Any], tutorial_goal: str) -> Dict[str, Any]:
        """
        Plan tutorial structure.
        
        Args:
            summarized_content: Dictionary with summarized PDF content from PDFSummarizer
            tutorial_goal: Goal or objective of the tutorial
            
        Returns:
            Dictionary with tutorial plan
        """
        if "error" in summarized_content:
            return {"error": f"Cannot plan tutorial due to summarization error: {summarized_content['error']}"}
        
        from langchain_ollama import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = ChatOllama(model=self.llm_model)
        
        # Format the summarized content for the LLM
        overall_summary = summarized_content.get("overall_summary", {})
        document_summary = overall_summary.get("document_summary", "")
        main_topics = ", ".join(overall_summary.get("main_topics", []))
        key_findings = ", ".join(overall_summary.get("key_findings", []))
        
        # Get information about images
        images_info = ""
        if summarized_content.get("images"):
            images_info = f"The document contains {len(summarized_content['images'])} images that can be used in the tutorial."
        
        # Create a prompt for tutorial planning
        planning_prompt = f"""
        I need to create a tutorial based on a PDF document. Here's the information:
        
        TUTORIAL GOAL: {tutorial_goal}
        
        DOCUMENT SUMMARY: {document_summary}
        
        MAIN TOPICS: {main_topics}
        
        KEY FINDINGS: {key_findings}
        
        {images_info}
        
        Based on this information, please help me plan a comprehensive tutorial structure.
        """
        
        response = llm.invoke([
            SystemMessage(content="""
            You are an expert tutorial planner. Create a detailed plan for a tutorial based on the provided document summary.
            The plan should include:
            1. A compelling title
            2. A brief introduction explaining what the tutorial will cover and who it's for
            3. A clear list of learning objectives
            4. A logical sequence of sections and subsections
            5. Key points to cover in each section
            6. Recommended examples, exercises, or hands-on activities
            7. A conclusion summarizing what was learned
            
            Structure your response as a JSON object with the following format:
            {
                "title": "Tutorial Title",
                "introduction": "Brief description of the tutorial",
                "target_audience": "Description of who this tutorial is for",
                "learning_objectives": ["Objective 1", "Objective 2", ...],
                "sections": [
                    {
                        "title": "Section Title",
                        "content_summary": "Brief summary of this section",
                        "key_points": ["Point 1", "Point 2", ...],
                        "subsections": [
                            {
                                "title": "Subsection Title",
                                "content_summary": "Brief summary of this subsection",
                                "key_points": ["Point 1", "Point 2", ...]
                            },
                            ...
                        ],
                        "examples": ["Example 1", "Example 2", ...],
                        "exercises": ["Exercise 1", "Exercise 2", ...]
                    },
                    ...
                ],
                "conclusion": "Summary of what was covered",
                "estimated_duration": "Estimated time to complete the tutorial"
            }
            """),
            HumanMessage(content=planning_prompt)
        ])
        
        # Extract JSON from response
        plan_text = response.content
        
        # Extract JSON part from potential non-JSON text
        json_match = re.search(r'{.*}', plan_text, re.DOTALL)
        if json_match:
            plan_text = json_match.group(0)
        
        try:
            import json
            plan_data = json.loads(plan_text)
            
            # Add additional metadata
            plan_data["pdf_filename"] = summarized_content.get("filename", "")
            plan_data["tutorial_goal"] = tutorial_goal
            
            return plan_data
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "error": "Could not parse tutorial plan",
                "raw_plan": plan_text,
                "title": "Untitled Tutorial",
                "introduction": "This tutorial will cover the main topics from the document.",
                "sections": [],
                "pdf_filename": summarized_content.get("filename", ""),
                "tutorial_goal": tutorial_goal
            }


class SimpleTutorialWriter:
    """Simple tool to write sections of a tutorial based on the plan."""
    
    def __init__(self, llm_model: str, rag_pipeline: Optional[RAGPipeline] = None):
        """
        Initialize tutorial writer.
        
        Args:
            llm_model: Name of the LLM model to use for writing
            rag_pipeline: RAG pipeline for retrieving relevant information
        """
        self.llm_model = llm_model
        self.rag_pipeline = rag_pipeline
    
    def write(
        self, 
        tutorial_plan: Dict[str, Any],
        pdf_content: Dict[str, Any],
        section_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Write tutorial section(s).
        
        Args:
            tutorial_plan: Dictionary with tutorial plan from TutorialPlanner
            pdf_content: Original PDF content from PDFExtractor
            section_index: Optional index of specific section to write (if None, write all)
            
        Returns:
            Dictionary with written tutorial sections
        """
        if "error" in tutorial_plan:
            return {"error": f"Cannot write tutorial due to planning error: {tutorial_plan['error']}"}
        
        from langchain_ollama import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = ChatOllama(model=self.llm_model)
        
        # Determine sections to write
        sections_to_write = []
        if section_index is not None:
            if section_index < 0 or section_index >= len(tutorial_plan.get("sections", [])):
                return {"error": f"Invalid section index: {section_index}"}
            sections_to_write = [tutorial_plan["sections"][section_index]]
        else:
            sections_to_write = tutorial_plan.get("sections", [])
        
        # Write each section
        written_sections = {}
        
        # Write introduction
        introduction_prompt = f"""
        Write the introduction for a tutorial titled '{tutorial_plan.get("title", "Tutorial")}'.
        
        TUTORIAL GOAL: {tutorial_plan.get("tutorial_goal", "")}
        
        INTRODUCTION SUMMARY: {tutorial_plan.get("introduction", "")}
        
        TARGET AUDIENCE: {tutorial_plan.get("target_audience", "")}
        
        LEARNING OBJECTIVES: {', '.join(tutorial_plan.get("learning_objectives", []))}
        """
        
        response = llm.invoke([
            SystemMessage(content="""
            You are an expert technical writer. Write an engaging introduction for a tutorial.
            The introduction should:
            1. Capture the reader's attention
            2. Clearly explain what the tutorial will cover
            3. Describe who the tutorial is for
            4. List the learning objectives
            5. Set expectations for what readers will accomplish
            
            Use a friendly, conversational tone but remain professional.
            Format your response using Markdown.
            """),
            HumanMessage(content=introduction_prompt)
        ])
        
        written_sections["introduction"] = response.content
        
        # Write each content section
        for idx, section in enumerate(sections_to_write):
            section_title = section.get("title", f"Section {idx+1}")
            section_summary = section.get("content_summary", "")
            key_points = section.get("key_points", [])
            subsections = section.get("subsections", [])
            examples = section.get("examples", [])
            exercises = section.get("exercises", [])
            
            # Find relevant content from PDF
            relevant_content = ""
            if self.rag_pipeline is not None:
                # Use RAG to find relevant content
                query = f"{section_title} {section_summary} {' '.join(key_points)}"
                try:
                    rag_results = self.rag_pipeline.query(query)
                    relevant_nodes = rag_results.get("source_nodes", [])
                    relevant_content = "\n\n".join([node.get("text", "") for node in relevant_nodes])
                except:
                    # Fallback to direct search in PDF content
                    relevant_content = self._find_relevant_content(pdf_content, section_title, key_points)
            else:
                # Direct search in PDF content
                relevant_content = self._find_relevant_content(pdf_content, section_title, key_points)
            
            # Create section prompt
            section_prompt = f"""
            Write the content for a section titled '{section_title}' in a tutorial.
            
            SECTION SUMMARY: {section_summary}
            
            KEY POINTS TO COVER:
            {', '.join(key_points)}
            
            SUBSECTIONS:
            {', '.join([sub.get("title", "") for sub in subsections])}
            
            EXAMPLES TO INCLUDE:
            {', '.join(examples)}
            
            EXERCISES TO INCLUDE:
            {', '.join(exercises)}
            
            RELEVANT CONTENT FROM SOURCE DOCUMENT:
            {relevant_content[:2000]}
            """
            
            response = llm.invoke([
                SystemMessage(content="""
                You are an expert technical writer. Write an informative, educational section for a tutorial.
                This section should:
                1. Start with a clear introduction to the section topic
                2. Cover all the key points thoroughly
                3. Include subsections with appropriate headings (use ## for subsection headings)
                4. Provide illustrative examples where appropriate
                5. Include code samples if relevant, properly formatted in Markdown
                6. End with exercises or activities if provided
                
                Use a friendly, educational tone. Explain concepts clearly.
                Make sure to format your response using Markdown.
                Use code blocks with language specifiers for any code examples.
                """),
                HumanMessage(content=section_prompt)
            ])
            
            written_sections[f"section_{idx}"] = {
                "title": section_title,
                "content": response.content,
                "subsections": []
            }
            
            # Write subsections if any
            for sub_idx, subsection in enumerate(subsections):
                subsection_title = subsection.get("title", f"Subsection {sub_idx+1}")
                subsection_summary = subsection.get("content_summary", "")
                subsection_key_points = subsection.get("key_points", [])
                
                # Subsection prompt
                subsection_prompt = f"""
                Write the content for a subsection titled '{subsection_title}' within the '{section_title}' section.
                
                SUBSECTION SUMMARY: {subsection_summary}
                
                KEY POINTS TO COVER:
                {', '.join(subsection_key_points)}
                
                RELEVANT CONTENT FROM SOURCE DOCUMENT:
                {relevant_content[:1500]}
                """
                
                response = llm.invoke([
                    SystemMessage(content="""
                    You are an expert technical writer. Write a focused subsection for a tutorial.
                    This subsection should:
                    1. Start with a brief introduction to the specific subtopic
                    2. Cover all the key points thoroughly
                    3. Provide concrete examples and explanations
                    4. Use code samples if relevant, properly formatted in Markdown
                    
                    Use a friendly, educational tone. Explain concepts clearly.
                    Make sure to format your response using Markdown.
                    Use code blocks with language specifiers for any code examples.
                    """),
                    HumanMessage(content=subsection_prompt)
                ])
                
                written_sections[f"section_{idx}"]["subsections"].append({
                    "title": subsection_title,
                    "content": response.content
                })
        
        # Write conclusion
        conclusion_prompt = f"""
        Write the conclusion for a tutorial titled '{tutorial_plan.get("title", "Tutorial")}'.
        
        TUTORIAL GOAL: {tutorial_plan.get("tutorial_goal", "")}
        
        LEARNING OBJECTIVES: {', '.join(tutorial_plan.get("learning_objectives", []))}
        
        CONCLUSION SUMMARY: {tutorial_plan.get("conclusion", "")}
        
        SECTIONS COVERED: {', '.join([section.get("title", "") for section in tutorial_plan.get("sections", [])])}
        """
        
        response = llm.invoke([
            SystemMessage(content="""
            You are an expert technical writer. Write a concise conclusion for a tutorial.
            The conclusion should:
            1. Summarize what was covered in the tutorial
            2. Reinforce key learning points
            3. Suggest next steps or further resources for readers
            4. End on an encouraging note
            
            Use a friendly, conversational tone.
            Format your response using Markdown.
            """),
            HumanMessage(content=conclusion_prompt)
        ])
        
        written_sections["conclusion"] = response.content
        
        return {
            "title": tutorial_plan.get("title", "Tutorial"),
            "sections": written_sections
        }
    
    def _find_relevant_content(
        self, pdf_content: Dict[str, Any], section_title: str, key_points: List[str]
    ) -> str:
        """
        Find relevant content in the PDF for a specific section.
        
        Args:
            pdf_content: PDF content from PDFExtractor
            section_title: Title of the section
            key_points: List of key points for the section
            
        Returns:
            String with relevant content
        """
        # Simple keyword-based search
        keywords = [section_title.lower()] + [point.lower() for point in key_points]
        
        relevant_text = []
        
        for page in pdf_content.get("pages", []):
            page_text = page.get("text", "").lower()
            
            # Score this page based on keyword matches
            score = sum(1 for keyword in keywords if keyword in page_text)
            
            if score > 0:
                relevant_text.append((score, page.get("text", "")))
        
        # Sort by relevance score and combine
        relevant_text.sort(reverse=True)
        return "\n\n".join([text for _, text in relevant_text[:3]])


class SimpleTutorialFormatter:
    """Simple tool to format the tutorial with proper markdown, code samples, and examples."""
    
    def __init__(self, llm_model: str):
        """
        Initialize tutorial formatter.
        
        Args:
            llm_model: Name of the LLM model to use for formatting
        """
        self.llm_model = llm_model
    
    def format(self, written_sections: Dict[str, Any], pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format tutorial.
        
        Args:
            written_sections: Dictionary with written tutorial sections from TutorialWriter
            pdf_content: Original PDF content from PDFExtractor
            
        Returns:
            Dictionary with formatted tutorial
        """
        from langchain_ollama import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = ChatOllama(model=self.llm_model)
        
        # Extract images information
        images = pdf_content.get("images", [])
        image_references = []
        
        for img in images:
            image_references.append({
                "page": img.get("page", 0),
                "path": img.get("path", ""),
                "description": f"Image from page {img.get('page', 0)}"
            })
        
        # Format the tutorial
        title = written_sections.get("title", "Tutorial")
        introduction = written_sections.get("sections", {}).get("introduction", "")
        conclusion = written_sections.get("sections", {}).get("conclusion", "")
        
        # Extract sections
        sections_content = []
        for key, value in written_sections.get("sections", {}).items():
            if key.startswith("section_"):
                section_title = value.get("title", "")
                section_content = value.get("content", "")
                
                # Process subsections
                subsections_content = []
                for subsection in value.get("subsections", []):
                    subsection_title = subsection.get("title", "")
                    subsection_content = subsection.get("content", "")
                    subsections_content.append(f"## {subsection_title}\n\n{subsection_content}")
                
                # Combine section with its subsections
                full_section = f"# {section_title}\n\n{section_content}\n\n"
                if subsections_content:
                    full_section += "\n\n".join(subsections_content)
                
                sections_content.append(full_section)
        
        # Create content for formatting
        space = "\n\n" 
        raw_tutorial = f"""
        # {title}
        
        ## Introduction
        
        {introduction}
        
        {space.join(sections_content)}
        
        ## Conclusion
        
        {conclusion}
        """
        
        # Image information
        images_info = ""
        if images:
            images_info = "The document contains these images that could be included:\n\n"
            for img in image_references:
                images_info += f"- Image from page {img['page']}: {img['path']}\n"
        
        # Format prompt
        format_prompt = f"""
        I have a raw tutorial that needs to be properly formatted with Markdown.
        
        {images_info}
        
        Here's the raw tutorial content:
        
        {raw_tutorial}
        
        Please format this into a well-structured tutorial with proper Markdown formatting.
        Add appropriate headers, code blocks, bullet points, and other formatting as needed.
        """
        
        response = llm.invoke([
            SystemMessage(content="""
            You are an expert technical writer and Markdown formatter. Format the provided tutorial content
            into a professional, well-structured document using Markdown.
            
            Your formatting should:
            1. Use appropriate heading levels (# for main title, ## for sections, ### for subsections)
            2. Format code examples using proper code blocks with language specifiers
            3. Use bullet points and numbered lists where appropriate
            4. Add emphasis (bold, italic) to highlight important concepts
            5. Include image references where mentioned
            6. Ensure consistent spacing and organization
            7. Add a table of contents at the beginning
            8. Format any tables properly using Markdown table syntax
            
            Make the document visually appealing and easy to read.
            """),
            HumanMessage(content=format_prompt)
        ])
        
        # Return the formatted tutorial
        return {
            "title": title,
            "formatted_content": response.content,
            "raw_content": raw_tutorial,
            "image_references": image_references
        }


class SimpleTutorialImprover:
    """Simple tool to identify and implement improvements to the tutorial."""
    
    def __init__(self, llm_model: str):
        """
        Initialize tutorial improver.
        
        Args:
            llm_model: Name of the LLM model to use for improvement
        """
        self.llm_model = llm_model
    
    def improve(self, formatted_tutorial: Dict[str, Any]) -> Dict[str, Any]:
        """
        Improve tutorial.
        
        Args:
            formatted_tutorial: Dictionary with formatted tutorial from TutorialFormatter
            
        Returns:
            Dictionary with improved tutorial and list of improvements
        """
        from langchain_ollama import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage
        import json
        
        llm = ChatOllama(model=self.llm_model)
        
        # Get the formatted content
        title = formatted_tutorial.get("title", "Tutorial")
        content = formatted_tutorial.get("formatted_content", "")
        
        # Analyze for improvements
        analysis_prompt = f"""
        Analyze this tutorial and identify ways to improve it:
        
        Title: {title}
        
        {content[:7500]}  # Using first part of content due to token limitations
        
        Please identify specific improvements that could make this tutorial more effective,
        clearer, more engaging, or more educational.
        """
        
        analysis_response = llm.invoke([
            SystemMessage(content="""
            You are an expert tutorial reviewer. Analyze the provided tutorial and identify specific improvements
            in the following categories:
            
            1. Clarity - Are concepts explained clearly? Are there confusing sections?
            2. Structure - Is the organization logical? Are transitions smooth?
            3. Examples - Are examples effective? Are more needed?
            4. Code samples - Are they clear, accurate, and well-commented?
            5. Visual elements - Would diagrams, tables, or images help?
            6. Engagement - Is the tone engaging? Are there interactive elements?
            7. Completeness - Are there missing explanations or steps?
            8. Accessibility - Is the content accessible to the target audience?
            
            For each suggested improvement, provide:
            - A clear description of the issue
            - A specific recommendation for addressing it
            - The section/location where the improvement should be made
            
            Format your response as a JSON object with an array of improvement objects:
            {
              "improvements": [
                {
                  "category": "Category name",
                  "issue": "Description of the issue",
                  "recommendation": "Specific recommendation",
                  "location": "Section or location in the tutorial"
                },
                ...
              ]
            }
            """),
            HumanMessage(content=analysis_prompt)
        ])
        
        # Extract JSON from response
        analysis_text = analysis_response.content
        
        # Extract JSON part from potential non-JSON text
        json_match = re.search(r'{.*}', analysis_text, re.DOTALL)
        if json_match:
            analysis_text = json_match.group(0)
        
        try:
            improvements_data = json.loads(analysis_text)
            improvements = improvements_data.get("improvements", [])
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            improvements = [
                {
                    "category": "General",
                    "issue": "Could not parse improvements JSON",
                    "recommendation": "Manual review needed",
                    "location": "Entire tutorial"
                }
            ]
        
        # Implement the improvements
        implementation_prompt = f"""
        Please improve this tutorial based on these suggestions:
        
        TUTORIAL:
        {content[:6000]}  # First part due to token limitations
        
        IMPROVEMENTS TO IMPLEMENT:
        {json.dumps(improvements, indent=2)}
        
        Apply these improvements to the tutorial content and return the improved version.
        """
        
        implementation_response = llm.invoke([
            SystemMessage(content="""
            You are an expert technical writer. Implement the suggested improvements to the tutorial.
            Make thoughtful changes that address each issue while maintaining the overall structure and flow.
            
            Your improvements should:
            1. Address each specific issue in the suggested location
            2. Maintain the tutorial's voice and style
            3. Enhance clarity, engagement, and educational value
            4. Preserve all valuable existing content
            5. Format the content properly using Markdown
            
            Return the improved tutorial in full.
            """),
            HumanMessage(content=implementation_prompt)
        ])
        
        # Return the improved tutorial and list of improvements
        return {
            "title": title,
            "original_content": content,
            "improved_content": implementation_response.content,
            "improvements": improvements
        }
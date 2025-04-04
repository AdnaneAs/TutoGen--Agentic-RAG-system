from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import os
import sys
import json
import re
import logging

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG pipeline
from rag.rag_pipeline import RAGPipeline
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AcademicWriterTool:
    """
    Tool for academic writing that processes tutorial sections according to a plan
    and uses a ReAct pattern for decision making.
    """
    
    def __init__(self, llm_model: str, rag_pipeline: Optional[RAGPipeline] = None):
        """
        Initialize the academic writer tool.
        
        Args:
            llm_model: Language model to use for writing
            rag_pipeline: Optional RAG pipeline for retrieving relevant content
        """
        self.llm_model = llm_model
        self.rag_pipeline = rag_pipeline
        self.llm = ChatOllama(model=llm_model)
    
    def retrieve_information(self, section_title: str, section_summary: str, key_points: List[str]) -> Dict[str, Any]:
        """
        Retrieve relevant information for a section using RAG pipeline.
        
        Args:
            section_title: Title of the section
            section_summary: Summary of the section
            key_points: Key points to cover in the section
            
        Returns:
            Dictionary containing retrieved information
        """
        if self.rag_pipeline is None:
            return {"error": "RAG pipeline not initialized"}
        
        try:
            # Construct a query from the section details
            query = f"{section_title} {section_summary} {' '.join(key_points)}"
            
            # Query the RAG pipeline
            rag_results = self.rag_pipeline.query(query, similarity_top_k=8)
            
            # Extract relevant content
            relevant_nodes = rag_results.get("source_nodes", [])
            relevant_content = []
            for node in relevant_nodes:
                source_info = f"Source: {node.get('metadata', {}).get('source', 'Unknown')}"
                page_info = f"Page: {node.get('metadata', {}).get('page_num', 'Unknown')}"
                content_text = node.get("text", "")
                
                # Include metadata about the source with the content
                annotated_content = f"{content_text}\n[{source_info}, {page_info}]"
                relevant_content.append(annotated_content)
            
            # Find relevant images and tables
            relevant_images = []
            relevant_tables = []
            
            # Look for images related to the section
            for node in relevant_nodes:
                metadata = node.get("metadata", {})
                if metadata.get("content_type") == "image":
                    relevant_images.append({
                        "path": metadata.get("image_path", ""),
                        "description": node.get("text", "").replace("Image Description: ", ""),
                        "page": metadata.get("page_num", 0),
                        "index": metadata.get("image_index", 0)
                    })
                elif metadata.get("content_type") == "table":
                    relevant_tables.append({
                        "path": metadata.get("table_path", ""),
                        "headers": metadata.get("headers", "").split(","),
                        "description": node.get("text", ""),
                        "page": metadata.get("page_num", 0),
                        "index": metadata.get("table_index", 0)
                    })
            
            return {
                "relevant_content": "\n\n".join(relevant_content),
                "relevant_images": relevant_images,
                "relevant_tables": relevant_tables,
                "query": query
            }
        except Exception as e:
            logger.error(f"Error retrieving information: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Error retrieving information: {str(e)}"}
    
    def write_section(self, section_data: Dict[str, Any], retrieved_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write a section based on section data and retrieved information.
        
        Args:
            section_data: Dictionary containing section details
            retrieved_info: Dictionary containing retrieved information
            
        Returns:
            Dictionary containing the written section
        """
        try:
            section_title = section_data.get("title", "")
            section_summary = section_data.get("content_summary", "")
            key_points = section_data.get("key_points", [])
            subsections = section_data.get("subsections", [])
            
            relevant_content = retrieved_info.get("relevant_content", "")
            relevant_images = retrieved_info.get("relevant_images", [])
            relevant_tables = retrieved_info.get("relevant_tables", [])
            
            # Create a prompt for writing the section
            image_info = ""
            if relevant_images:
                image_info = "Available relevant images:\n"
                for i, img in enumerate(relevant_images):
                    image_info += f"{i+1}. {img.get('description', 'No description')} (Page {img.get('page', 'unknown')})\n"
            
            table_info = ""
            if relevant_tables:
                table_info = "Available relevant tables:\n"
                for i, table in enumerate(relevant_tables):
                    headers = ', '.join(table.get('headers', []))
                    table_info += f"{i+1}. Table with headers: {headers} (Page {table.get('page', 'unknown')})\n"
            
            # Create a writing prompt
            writing_prompt = f"""
            Write a comprehensive academic section titled '{section_title}' for a tutorial.
            
            SECTION SUMMARY: {section_summary}
            
            KEY POINTS TO COVER (elaborate extensively on each):
            {', '.join(key_points)}
            
            SUBSECTIONS TO DEVELOP (each should be substantial with multiple paragraphs):
            {', '.join([sub.get("title", "") for sub in subsections])}
            
            {image_info}
            
            {table_info}
            
            RELEVANT CONTENT FROM SOURCE DOCUMENT:
            {relevant_content[:4000]}
            
            This section should be extensive (at least 1500-2000 words) with multiple subsections, each containing 
            several well-developed paragraphs. Use academic language and provide thorough explanations of all concepts.
            Include theoretical foundations, practical applications, and connect to broader academic contexts.
            
            When referencing images, use the format: ![Image description](path/to/image)
            When including tables, provide detailed analysis of the data shown.
            
            Format your writing with proper Markdown headings, lists, and emphasis.
            """
            
            # Use the LLM to write the section
            response = self.llm.invoke([
                SystemMessage(content="""
                You are an expert Academic writer creating comprehensive educational content. Write an extensive, 
                in-depth section for an academic tutorial.
                
                Your section MUST:
                1. Begin with a substantive introduction (3-4 paragraphs) that thoroughly frames the section topic
                2. Cover each key point with exceptional depth, devoting multiple paragraphs to each concept
                3. Provide theoretical foundations and academic context for all topics covered
                4. Include rich, nuanced explanations that demonstrate expert understanding
                5. Develop multiple detailed subsections (at least 3-5 paragraphs each)
                6. Provide comprehensive examples with thorough explanations
                7. Include detailed code samples if relevant, with extensive comments explaining each component
                8. Reference and explain tables in depth, analyzing their significance
                9. Connect concepts to broader academic literature and practical applications
                10. End with substantive exercises or activities with detailed instructions
                11. Use precise academic terminology and sophisticated language throughout
                
                When referencing images:
                - Use descriptive captions that explain the significance
                - Format as: ![Figure X: Description](path/to/image)
                - Reference figures in the text (e.g., "As shown in Figure X...")
                
                When referencing tables:
                - Analyze the data thoroughly with multiple paragraphs of explanation
                - Explain the theoretical significance of the data
                - Connect the table content to the broader concepts being taught
                
                Use a formal academic tone. Each concept should be explained with exceptional depth and clarity.
                Make sure to format your response using sophisticated Markdown.
                Use code blocks with language specifiers for any code examples.
                """),
                HumanMessage(content=writing_prompt)
            ])
            
            # Process the response to include proper image references
            content = response.content
            
            # Replace generic image references with actual paths
            for i, img in enumerate(relevant_images):
                img_path = img.get('path', '')
                if img_path and f"Figure {i+1}" in content:
                    pattern = f"(Figure {i+1}:.*?)(\n\n|\n#)"
                    replacement = f"![Figure {i+1}: {img.get('description', '')}]({img_path})\n\n"
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            return {
                "title": section_title,
                "content": content,
                "images_used": [img.get('path') for img in relevant_images if img.get('path') in content],
                "tables_used": [table.get('path') for table in relevant_tables if table.get('path') in content]
            }
            
        except Exception as e:
            logger.error(f"Error writing section: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Error writing section: {str(e)}"}
    
    def write_introduction(self, tutorial_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write introduction for the tutorial.
        
        Args:
            tutorial_plan: Dictionary containing tutorial plan
            
        Returns:
            Dictionary containing the introduction
        """
        try:
            title = tutorial_plan.get("title", "Tutorial")
            introduction_summary = tutorial_plan.get("introduction", "")
            tutorial_goal = tutorial_plan.get("tutorial_goal", "")
            target_audience = tutorial_plan.get("target_audience", "")
            learning_objectives = tutorial_plan.get("learning_objectives", [])
            
            introduction_prompt = f"""
            Write a comprehensive and detailed introduction for a tutorial titled '{title}'.
            
            TUTORIAL GOAL: {tutorial_goal}
            
            INTRODUCTION SUMMARY: {introduction_summary}
            
            TARGET AUDIENCE: {target_audience}
            
            LEARNING OBJECTIVES: {', '.join(learning_objectives)}
            
            The introduction should be at least 4-5 well-developed paragraphs that thoroughly introduce the topic,
            establish context, and motivate the reader's interest. Include relevant background information,
            current trends in the field, and the significance of the topic. Make connections to academic research
            or industry applications where appropriate.
            """
            
            response = self.llm.invoke([
                SystemMessage(content="""
                You are an expert Academic writer. Write an engaging, comprehensive introduction for a tutorial.
                
                Your introduction MUST:
                1. Be extensive (at least 4-5 well-developed paragraphs)
                2. Provide rich background context and establish the academic importance of the topic
                3. Clearly articulate what the tutorial will cover with significant detail
                4. Describe who the tutorial is for with nuanced explanations of prior knowledge expected
                5. List and elaborate on each learning objective with thorough explanations
                6. Establish the theoretical foundations that underpin the tutorial content
                7. Connect the tutorial content to broader academic or practical contexts
                8. Include relevant statistics, research findings, or industry trends when applicable
                9. Set detailed expectations for what readers will accomplish
                10. Use formal academic language and precise terminology
                
                Use a professional academic tone throughout. Make every paragraph substantive and information-rich.
                Format your response using Markdown, with appropriate headings, emphasis, and structure.
                """),
                HumanMessage(content=introduction_prompt)
            ])
            
            return {
                "title": "Introduction",
                "content": response.content
            }
            
        except Exception as e:
            logger.error(f"Error writing introduction: {str(e)}")
            return {"error": f"Error writing introduction: {str(e)}"}
    
    def write_conclusion(self, tutorial_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write conclusion for the tutorial.
        
        Args:
            tutorial_plan: Dictionary containing tutorial plan
            
        Returns:
            Dictionary containing the conclusion
        """
        try:
            title = tutorial_plan.get("title", "Tutorial")
            tutorial_goal = tutorial_plan.get("tutorial_goal", "")
            learning_objectives = tutorial_plan.get("learning_objectives", [])
            conclusion_summary = tutorial_plan.get("conclusion", "")
            sections = tutorial_plan.get("sections", [])
            
            conclusion_prompt = f"""
            Write a comprehensive, academic conclusion for the tutorial titled '{title}'.
            
            TUTORIAL GOAL: {tutorial_goal}
            
            LEARNING OBJECTIVES: {', '.join(learning_objectives)}
            
            CONCLUSION SUMMARY: {conclusion_summary}
            
            SECTIONS COVERED: {', '.join([section.get("title", "") for section in sections])}
            
            This conclusion should be substantial (at least 3-4 well-developed paragraphs) and should synthesize the key
            concepts covered throughout the tutorial. It should also highlight the significance of the material in a broader
            academic or practical context. Include suggested paths for further exploration or research in the field.
            """
            
            response = self.llm.invoke([
                SystemMessage(content="""
                You are an expert Academic writer. Write a comprehensive, scholarly conclusion for a tutorial.
                
                Your conclusion MUST:
                1. Be substantial (at least 3-4 well-developed paragraphs)
                2. Thoroughly summarize and synthesize the key concepts covered in the tutorial
                3. Highlight the theoretical significance of the material presented
                4. Connect the concepts to broader academic contexts or practical applications
                5. Reinforce key learning points with sophisticated analysis
                6. Suggest multiple specific next steps for further learning or research
                7. Provide recommendations for additional resources with brief explanations
                8. End with a thought-provoking statement that emphasizes the importance of the topic
                9. Use formal academic language and precise terminology
                
                Use a professional, scholarly tone throughout. Make every paragraph substantive and information-rich.
                Format your response using sophisticated Markdown, with appropriate emphasis and structure.
                """),
                HumanMessage(content=conclusion_prompt)
            ])
            
            return {
                "title": "Conclusion",
                "content": response.content
            }
            
        except Exception as e:
            logger.error(f"Error writing conclusion: {str(e)}")
            return {"error": f"Error writing conclusion: {str(e)}"}
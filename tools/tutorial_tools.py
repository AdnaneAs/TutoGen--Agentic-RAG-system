from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import os
import sys
import json
import re

# Import RAG pipeline (using relative imports)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.rag_pipeline import RAGPipeline
from embedding.image_embedding import get_image_embedding_model
from embedding.text_embedding import get_text_embedding_model
from embedding.tables_embedding import get_table_embedding_model
from llama_index.core.schema import Document, ImageDocument
from llama_index.core import Settings
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage


class SimpleTutorialPlanner:
    def __init__(self, llm_model: str):
        self.llm_model = llm_model
    
    def plan(self, summarized_content: Dict[str, Any], tutorial_goal: str) -> Dict[str, Any]:
        if summarized_content is None:
            return {"error": "Cannot plan tutorial: summarized_content is None"}
            
        if "error" in summarized_content:
            return {"error": f"Cannot plan tutorial due to summarization error: {summarized_content['error']}"}
        
        llm = ChatOllama(model=self.llm_model)
        
        overall_summary = summarized_content.get("overall_summary", {})
        document_summary = overall_summary.get("document_summary", "")
        main_topics = ", ".join(overall_summary.get("main_topics", []))
        key_findings = ", ".join(overall_summary.get("key_findings", []))
        
        methodologies = overall_summary.get("methodologies", [])
        theoretical_frameworks = overall_summary.get("theoretical_frameworks", [])
        knowledge_domains = overall_summary.get("knowledge_domains", [])
        target_audience = overall_summary.get("target_audience", "")
        
        technical_terms = []
        for page in summarized_content.get("page_summaries", []):
            if "technical_terms" in page:
                technical_terms.extend(page.get("technical_terms", []))
        
        images_info = ""
        if summarized_content.get("images"):
            images_info = f"The document contains {len(summarized_content['images'])} images that can be used in the tutorial."
        
        tables_info = ""
        if summarized_content.get("tables"):
            tables_info = f"The document contains {len(summarized_content['tables'])} tables that can be used in the tutorial."
            if "tables_summary" in summarized_content and "summary" in summarized_content["tables_summary"]:
                tables_info += f"\nTable summary: {summarized_content['tables_summary']['summary']}"
        
        planning_prompt = f"""
        I need to create a comprehensive academic tutorial based on a PDF document. Here's the information:
        
        TUTORIAL GOAL: {tutorial_goal}
        
        DOCUMENT SUMMARY: {document_summary}
        
        MAIN TOPICS: {main_topics}
        
        KEY FINDINGS: {key_findings}
        
        {"THEORETICAL FRAMEWORKS: " + ", ".join(theoretical_frameworks) if theoretical_frameworks else ""}
        
        {"METHODOLOGIES: " + ", ".join(methodologies) if methodologies else ""}
        
        {"KNOWLEDGE DOMAINS: " + ", ".join(knowledge_domains) if knowledge_domains else ""}
        
        {"TARGET AUDIENCE: " + target_audience if target_audience else ""}
        
        {images_info}
        
        {tables_info}
        
        Based on this information, please develop a structured, pedagogically sound tutorial plan with CLEAR HEADINGS and WELL-DEFINED SECTIONS that effectively teaches these concepts. 
        The tutorial structure should be concrete with specific section headings (not generic ones like "Main Topic 1").
        Each section should have a clear purpose and progression to facilitate learning.
        """
        
        response = llm.invoke([
            SystemMessage(content="""
            You are an expert instructional designer with extensive experience creating academic tutorials.
            
            Create a detailed, pedagogically sound tutorial plan based on the document summary provided. Your plan should follow best practices in educational design:
            
            1. SCAFFOLDED LEARNING STRUCTURE
            - Develop a logical progression from foundational to advanced concepts
            - Ensure each section builds upon previous knowledge
            - Create explicit connections between theory and practice
            
            2. ACTIVE LEARNING COMPONENTS
            - Incorporate varied engagement strategies (discussions, reflections, exercises)
            - Design hands-on activities that reinforce key concepts
            - Create opportunities for knowledge application and synthesis
            
            3. ASSESSMENT & FEEDBACK MECHANISMS
            - Include formative assessment opportunities throughout
            - Design exercises that test both conceptual understanding and practical application
            - Provide clear success criteria for learning activities
            
            4. MULTIMODAL CONTENT INTEGRATION
            - Strategically incorporate available visual elements (images, tables, diagrams)
            - Use diverse content formats to address different learning styles
            - Balance theoretical explanations with concrete examples
            
            Structure your response as a complete JSON object with the following format:
            {
                "title": "Clear, engaging title that accurately reflects content",
                "introduction": "Detailed introduction establishing context and relevance",
                "target_audience": "Specific description of intended learners and required prior knowledge",
                "learning_objectives": ["Measurable objective 1", "Measurable objective 2", ...],
                "prerequisites": ["Prerequisite 1", "Prerequisite 2", ...],
                "recommended_duration": "Estimated time for completion",
                "materials_needed": ["Material 1", "Material 2", ...],
                "sections": [
                    {
                        "title": "Section title",
                        "type": "One of: foundational, theoretical, practical, advanced, case_study",
                        "learning_goals": ["Goal 1", "Goal 2", ...],
                        "content_summary": "Detailed description of what this section covers",
                        "key_points": ["Point 1", "Point 2", ...],
                        "activities": ["Activity description 1", "Activity description 2", ...],
                        "subsections": [
                            {
                                "title": "Subsection title",
                                "content_summary": "Detailed description of subsection content",
                                "key_points": ["Point 1", "Point 2", ...],
                                "activities": ["Activity description", ...]
                            },
                            ...
                        ],
                        "examples": ["Example 1", "Example 2", ...],
                        "exercises": ["Exercise 1", "Exercise 2", ...],
                        "visual_aids": ["Visual aid 1", "Visual aid 2", ...],
                        "assessment": "Description of how learning will be assessed in this section"
                    },
                    ...
                ],
                "conclusion": "Summary of what was covered and next steps",
                "further_resources": ["Resource 1", "Resource 2", ...],
                "glossary_terms": [{"term": "Term 1", "definition": "Definition 1"}, ...]
            }
            
            Your plan should be detailed, comprehensive, and academically rigorous while remaining accessible and engaging.
            """),
            HumanMessage(content=planning_prompt)
        ])
        
        plan_text = response.content
        
        json_match = re.search(r'{.*}', plan_text, re.DOTALL)
        if json_match:
            plan_text = json_match.group(0)
        
        try:
            plan_data = json.loads(plan_text)
            
            plan_data["pdf_filename"] = summarized_content.get("filename", "")
            plan_data["tutorial_goal"] = tutorial_goal
            
            return plan_data
        except json.JSONDecodeError:
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
    def __init__(self, llm_model: str, rag_pipeline: Optional[RAGPipeline] = None):
        self.llm_model = llm_model
        self.rag_pipeline = rag_pipeline
    
    def write(
        self, 
        tutorial_plan: Dict[str, Any],
        pdf_content: Dict[str, Any],
        section_index: Optional[int] = None
    ) -> Dict[str, Any]:
        if "error" in tutorial_plan:
            return {"error": f"Cannot write tutorial due to planning error: {tutorial_plan['error']}"}
        
        llm = ChatOllama(model=self.llm_model)
        
        sections_to_write = []
        if section_index is not None:
            if section_index < 0 or section_index >= len(tutorial_plan.get("sections", [])):
                return {"error": f"Invalid section index: {section_index}"}
            sections_to_write = [tutorial_plan["sections"][section_index]]
        else:
            sections_to_write = tutorial_plan.get("sections", [])
        
        table_info = ""
        if pdf_content.get("tables"):
            table_info = f"The document contains {len(pdf_content['tables'])} tables.\n"
            for table in pdf_content.get("tables", []):
                table_info += f"- Table on page {table.get('page', 'N/A')} with {table.get('rows', 0)} rows and {table.get('columns', 0)} columns.\n"
                if table.get("headers"):
                    table_info += f"  Headers: {', '.join(table.get('headers', []))}\n"
        
        written_sections = {}
        
        introduction_prompt = f"""
        Write a comprehensive and detailed introduction for a tutorial titled '{tutorial_plan.get("title", "Tutorial")}'.
        
        TUTORIAL GOAL: {tutorial_plan.get("tutorial_goal", "")}
        
        INTRODUCTION SUMMARY: {tutorial_plan.get("introduction", "")}
        
        TARGET AUDIENCE: {tutorial_plan.get("target_audience", "")}
        
        LEARNING OBJECTIVES: {', '.join(tutorial_plan.get("learning_objectives", []))}
        
        DOCUMENT INFORMATION:
        {table_info}
        
        The introduction should be at least 4-5 well-developed paragraphs that thoroughly introduce the topic,
        establish context, and motivate the reader's interest. Include relevant background information,
        current trends in the field, and the significance of the topic. Make connections to academic research
        or industry applications where appropriate.
        """
        
        response = llm.invoke([
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
            Aim for approximately 800-1000 words for the introduction.
            """),
            HumanMessage(content=introduction_prompt)
        ])
        
        written_sections["introduction"] = response.content
        
        for idx, section in enumerate(sections_to_write):
            section_title = section.get("title", f"Section {idx+1}")
            section_summary = section.get("content_summary", "")
            key_points = section.get("key_points", [])
            subsections = section.get("subsections", [])
            examples = section.get("examples", [])
            exercises = section.get("exercises", [])
            
            relevant_content = ""
            relevant_tables = []
            
            if self.rag_pipeline is not None:
                query = f"{section_title} {section_summary} {' '.join(key_points)}"
                try:
                    rag_results = self.rag_pipeline.query(query, top_k=10)
                    relevant_nodes = rag_results.get("source_nodes", [])
                    relevant_content = "\n\n".join([node.get("text", "") for node in relevant_nodes])
                except:
                    relevant_content = self._find_relevant_content(pdf_content, section_title, key_points, max_pages=8)
            else:
                relevant_content = self._find_relevant_content(pdf_content, section_title, key_points, max_pages=8)
            
            for table in pdf_content.get("tables", []):
                headers = " ".join(table.get("headers", []))
                table_content = str(table.get("preview", []))
                
                if any(keyword.lower() in headers.lower() or keyword.lower() in table_content.lower() 
                       for keyword in [section_title.lower()] + [kp.lower() for kp in key_points]):
                    relevant_tables.append(table)
            
            section_table_info = ""
            if relevant_tables:
                section_table_info = f"RELEVANT TABLES:\n"
                for table in relevant_tables:
                    section_table_info += f"- Table on page {table.get('page', 'N/A')} with headers: {', '.join(table.get('headers', []))}\n"
                    
                    if table.get("preview"):
                        section_table_info += "  Preview data:\n"
                        for row in table.get("preview", [])[:3]:
                            row_str = "  - " + ", ".join([f"{k}: {v}" for k, v in row.items()]) + "\n"
                            section_table_info += row_str
            
            section_prompt = f"""
            Write a comprehensive, in-depth academic section titled '{section_title}' for a tutorial.
            
            SECTION SUMMARY: {section_summary}
            
            KEY POINTS TO COVER (elaborate extensively on each):
            {', '.join(key_points)}
            
            SUBSECTIONS TO DEVELOP (each should be substantial with multiple paragraphs):
            {', '.join([sub.get("title", "") for sub in subsections])}
            
            EXAMPLES TO INCLUDE (provide detailed explanations for each):
            {', '.join(examples)}
            
            EXERCISES TO INCLUDE (develop thoroughly with step-by-step instructions):
            {', '.join(exercises)}
            
            {section_table_info}
            
            RELEVANT CONTENT FROM SOURCE DOCUMENT:
            {relevant_content[:4000]}
            
            This section should be extensive (at least 1500-2000 words) with multiple subsections, each containing 
            several well-developed paragraphs. Use academic language and provide thorough explanations of all concepts. 
            Include theoretical foundations, practical applications, and connect to broader academic contexts.
            """
            
            response = llm.invoke([
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
                
                When referencing tables:
                - Analyze the data thoroughly with multiple paragraphs of explanation
                - Explain the theoretical significance of the data
                - Connect the table content to the broader concepts being taught
                - Format table references using proper Markdown table syntax
                
                Use a formal academic tone. Each concept should be explained with exceptional depth and clarity.
                Make sure to format your response using sophisticated Markdown.
                Use code blocks with language specifiers for any code examples.
                
                Aim for approximately 1500-2000 words for this section, with substantial development of all concepts.
                """),
                HumanMessage(content=section_prompt)
            ])
            
            written_sections[f"section_{idx}"] = {
                "title": section_title,
                "content": response.content,
                "subsections": []
            }
            
            for sub_idx, subsection in enumerate(subsections):
                subsection_title = subsection.get("title", f"Subsection {sub_idx+1}")
                subsection_summary = subsection.get("content_summary", "")
                subsection_key_points = subsection.get("key_points", [])
                
                subsection_relevant_tables = []
                for table in pdf_content.get("tables", []):
                    headers = " ".join(table.get("headers", []))
                    table_content = str(table.get("preview", []))
                    
                    if any(keyword.lower() in headers.lower() or keyword.lower() in table_content.lower()
                          for keyword in [subsection_title.lower()] + [kp.lower() for kp in subsection_key_points]):
                        subsection_relevant_tables.append(table)
                
                subsection_table_info = ""
                if subsection_relevant_tables:
                    subsection_table_info = f"RELEVANT TABLES FOR SUBSECTION:\n"
                    for table in subsection_relevant_tables:
                        subsection_table_info += f"- Table on page {table.get('page', 'N/A')} with headers: {', '.join(table.get('headers', []))}\n"
                        if table.get("preview"):
                            subsection_table_info += "  Preview data:\n"
                            for row in table.get("preview", [])[:3]:
                                row_str = "  - " + ", ".join([f"{k}: {v}" for k, v in row.items()]) + "\n"
                                subsection_table_info += row_str
                
                subsection_prompt = f"""
                Write a comprehensive and detailed subsection titled '{subsection_title}' within the '{section_title}' section.
                
                SUBSECTION SUMMARY: {subsection_summary}
                
                KEY POINTS TO COVER (elaborate extensively on each):
                {', '.join(subsection_key_points)}
                
                {subsection_table_info}
                
                RELEVANT CONTENT FROM SOURCE DOCUMENT:
                {relevant_content[:3000]}
                
                This subsection should be extensive (at least 800-1000 words) with multiple well-developed paragraphs.
                Provide in-depth explanations of all concepts, theoretical foundations, and practical applications.
                Include detailed examples and connect to broader academic contexts where appropriate. Use precise
                academic terminology and sophisticated language throughout.
                """
                
                response = llm.invoke([
                    SystemMessage(content="""
                    You are an expert Academic writer. Write a comprehensive, detailed subsection for an academic tutorial.
                    
                    Your subsection MUST:
                    1. Begin with a substantive introduction (2-3 paragraphs) that thoroughly frames the subtopic
                    2. Cover each key point with exceptional depth, devoting multiple paragraphs to each
                    3. Provide theoretical foundations and academic context for all topics covered
                    4. Include rich, nuanced explanations that demonstrate expert understanding
                    5. Develop at least 3-4 well-developed paragraphs for each major concept
                    6. Provide multiple detailed examples with thorough explanations
                    7. Include detailed code samples if relevant, with extensive comments explaining each component
                    8. Reference and explain tables in depth, analyzing their significance
                    9. Connect concepts to broader academic literature and practical applications
                    10. Use precise academic terminology and sophisticated language throughout
                    
                    When referencing tables:
                    - Analyze the data thoroughly with multiple paragraphs of explanation
                    - Explain the theoretical significance of the data
                    - Connect the table content to the broader concepts being taught
                    - Format table references using proper Markdown table syntax
                    
                    Use a formal academic tone. Each concept should be explained with exceptional depth and clarity.
                    Make sure to format your response using sophisticated Markdown.
                    Use code blocks with language specifiers for any code examples.
                    
                    Aim for approximately 800-1000 words for this subsection, with substantial development of all concepts.
                    """),
                    HumanMessage(content=subsection_prompt)
                ])
                
                written_sections[f"section_{idx}"]["subsections"].append({
                    "title": subsection_title,
                    "content": response.content
                })
        
        conclusion_prompt = f"""
        Write a comprehensive, academic conclusion for the tutorial titled '{tutorial_plan.get("title", "Tutorial")}'.
        
        TUTORIAL GOAL: {tutorial_plan.get("tutorial_goal", "")}
        
        LEARNING OBJECTIVES: {', '.join(tutorial_plan.get("learning_objectives", []))}
        
        CONCLUSION SUMMARY: {tutorial_plan.get("conclusion", "")}
        
        SECTIONS COVERED: {', '.join([section.get("title", "") for section in tutorial_plan.get("sections", [])])}
        
        This conclusion should be substantial (at least 3-4 well-developed paragraphs) and should synthesize the key
        concepts covered throughout the tutorial. It should also highlight the significance of the material in a broader
        academic or practical context. Include suggested paths for further exploration or research in the field.
        """
        
        response = llm.invoke([
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
            Aim for approximately 500-700 words for the conclusion.
            """),
            HumanMessage(content=conclusion_prompt)
        ])
        
        written_sections["conclusion"] = response.content
        
        return {
            "title": tutorial_plan.get("title", "Tutorial"),
            "sections": written_sections
        }
    
    def _find_relevant_content(
        self, pdf_content: Dict[str, Any], section_title: str, key_points: List[str], max_pages: int = 8
    ) -> str:
        primary_keywords = [section_title.lower()]
        secondary_keywords = [point.lower() for point in key_points]
        
        keyword_variants = []
        for kw in primary_keywords + secondary_keywords:
            keyword_variants.append(kw)
            keyword_variants.extend([w for w in kw.split() if len(w) > 3])
        
        relevant_text = []
        
        for page in pdf_content.get("pages", []):
            page_text = page.get("text", "").lower()
            
            primary_score = sum(3 for keyword in primary_keywords if keyword in page_text)
            secondary_score = sum(1 for keyword in secondary_keywords if keyword in page_text)
            variant_score = sum(0.5 for keyword in keyword_variants if keyword in page_text)
            
            total_score = primary_score + secondary_score + variant_score
            
            if total_score > 0:
                relevant_text.append((total_score, page.get("text", "")))
        
        relevant_text.sort(reverse=True)
        return "\n\n".join([text for _, text in relevant_text[:max_pages]])
    
class SimpleTutorialFormatter:
    def __init__(self, llm_model: str):
        self.llm_model = llm_model
    
    def format(self, written_sections: Dict[str, Any], pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        from langchain_ollama import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage
        import re
        
        llm = ChatOllama(model=self.llm_model)
        
        title = written_sections.get("title", "Tutorial")
        introduction = written_sections.get("sections", {}).get("introduction", "")
        conclusion = written_sections.get("sections", {}).get("conclusion", "")
        
        images = pdf_content.get("images", [])
        image_references = []
        
        for img_idx, img in enumerate(images):
            context_text = ""
            img_page = img.get("page", 0)
            
            for page in pdf_content.get("pages", []):
                if page.get("page_num") == img_page:
                    page_text = page.get("text", "")
                    context_text = self._extract_image_context(page_text)
                    break
            
            image_description = self._generate_image_description(
                f"Image {img_idx+1} from page {img.get('page', 0)}", 
                context_text,
                img.get("path", "")
            )
            
            image_references.append({
                "page": img.get("page", 0),
                "path": img.get("path", ""),
                "description": image_description,
                "context": context_text[:200] + "..." if len(context_text) > 200 else context_text,
                "index": img_idx + 1
            })
        
        tables = pdf_content.get("tables", [])
        table_references = []
        
        for table_idx, table in enumerate(tables):
            table_context = ""
            table_page = table.get("page", 0)
            
            for page in pdf_content.get("pages", []):
                if page.get("page_num") == table_page:
                    page_text = page.get("text", "")
                    table_context = self._extract_table_context(page_text)
                    break
            
            table_description = self._generate_table_description(
                table.get("headers", []),
                f"Table from page {table.get('page', 0)} with {table.get('rows', 0)} rows and {table.get('columns', 0)} columns",
                table_context
            )
            
            table_references.append({
                "page": table.get("page", 0),
                "path": table.get("csv_path", ""),
                "rows": table.get("rows", 0),
                "columns": table.get("columns", 0),
                "headers": table.get("headers", []),
                "description": table_description,
                "context": table_context[:200] + "..." if len(table_context) > 200 else table_context,
                "preview": table.get("preview", []),
                "index": table_idx + 1
            })
        
        sections_content = []
        for key, value in written_sections.get("sections", {}).items():
            if key.startswith("section_"):
                section_title = value.get("title", "")
                section_content = value.get("content", "")
                
                subsections_content = []
                for sub_idx, subsection in enumerate(value.get("subsections", [])):
                    subsection_title = subsection.get("title", "")
                    subsection_content = subsection.get("content", "")
                    
                    formatted_subsection = f"## {subsection_title}\n\n{subsection_content}"
                    
                    if sub_idx == 0:
                        formatted_subsection = f"{formatted_subsection}\n\n"
                    
                    subsections_content.append(formatted_subsection)
                
                full_section = f"# {section_title}\n\n{section_content}\n\n"
                if subsections_content:
                    full_section += "\n\n".join(subsections_content)
                
                sections_content.append(full_section)
        
        current_date = self._get_current_date()
        
        citation_info = f"""
## How to Cite This Tutorial

{self._generate_citation(title, current_date)}
        """
        
        front_matter = self._generate_front_matter(title, written_sections, current_date)
        
        space = "\n\n" 
        raw_tutorial = f"""
{front_matter}

# {title}

## Introduction

{introduction}

{space.join(sections_content)}

## Conclusion

{conclusion}

{citation_info}
        """
        
        images_info = ""
        if images:
            images_info = "The document contains these images that should be integrated into the tutorial:\n\n"
            for img in image_references:
                images_info += f"- Figure {img['index']}: {img['description']}\n"
                images_info += f"  Path: {img['path']}\n"
                images_info += f"  Context: {img['context']}\n\n"
        
        tables_info = ""
        if tables:
            tables_info = "The document contains these tables that should be integrated into the tutorial:\n\n"
            for table in table_references:
                tables_info += f"- Table {table['index']}: {table['description']}\n"
                tables_info += f"  Headers: {', '.join(table['headers'])}\n"
                
                if table.get("preview"):
                    tables_info += "  Preview data (first 3 rows):\n"
                    for row_idx, row in enumerate(table.get("preview", [])[:3]):
                        tables_info += f"    Row {row_idx + 1}: {row}\n"
                tables_info += "\n"
        
        format_prompt = f"""
        I have a raw academic tutorial that needs to be formatted with professional, publication-quality Markdown.
        
        TUTORIAL METADATA:
        - Title: {title}
        - Date: {current_date}
        
        IMAGE INFORMATION:
        {images_info}
        
        TABLE INFORMATION:
        {tables_info}
        
        RAW TUTORIAL CONTENT:
        
        {raw_tutorial}
        
        Please format this into a sophisticated academic tutorial with high-quality Markdown formatting that would be
        suitable for publication. The tutorial should have the following features:
        
        1. A visually appealing layout with consistent formatting throughout
        2. A comprehensive table of contents at the beginning
        3. Properly numbered sections and subsections
        4. Properly integrated and referenced figures with captions
        5. Properly formatted tables with captions
        6. Academic citation formatting (IEEE style)
        7. Highlighted key concepts and terminology
        8. Properly formatted code blocks with syntax highlighting
        9. Callout boxes for important notes, warnings, or tips
        10. Enhanced typography (proper use of emphasis, quotes, etc.)
        11. Well-structured lists and definitions
        12. Academic formatting conventions throughout
        
        For any tables, convert the text description into properly formatted Markdown tables.
        For any images, include proper figure captions and references in the text.
        """
        
        response = llm.invoke([
            SystemMessage(content="""
            You are an expert academic publication formatter with extensive experience in creating 
            professional Markdown documents. Format the provided tutorial content into a publication-quality
            academic document using enhanced Markdown.
            
            Your formatting MUST include:
            
            1. DOCUMENT STRUCTURE:
               - Title page with complete metadata
               - Abstract or overview section
               - Detailed table of contents with page section references
               - Properly numbered and hierarchical headings (following academic conventions)
               - Logical section breaks with appropriate spacing
               - Bibliography/references section with proper academic citations
            
            2. VISUAL FORMATTING:
               - Consistent and professional typography
               - Proper indentation and alignment of elements
               - Visual separation between distinct sections
               - Highlighted boxes for definitions or important concepts
               - Callout boxes for notes, warnings, or tips (using > for blockquotes with emojis)
               - Advanced table formatting with merged cells where appropriate
            
            3. ACADEMIC ELEMENTS:
               - Properly numbered figures with descriptive captions
               - Properly numbered tables with descriptive captions
               - In-text citations using IEEE format [1]
               - Footnotes for additional information where relevant
               - Glossary entries for specialized terminology
            
            4. CODE AND TECHNICAL CONTENT:
               - Syntax-highlighted code blocks with language specifiers
               - Inline code formatting for variable names, functions, etc.
               - Annotations or comments in code blocks
               - Command-line examples with proper formatting
            
            5. LISTS AND ORGANIZATIONAL ELEMENTS:
               - Hierarchical numbered lists for procedures
               - Bullet lists for non-sequential items
               - Definition lists for terminology
               - Task lists for exercises or activities
            
            6. TABLES:
               - Format all tables using proper Markdown table syntax
               - Include table numbers and captions
               - Use formatting like this:
                 | Header1 | Header2 | Header3 |
                 |:--------|:-------:|--------:|
                 | Left    | Center  | Right   |
            
            7. IMAGES AND FIGURES:
               - Include figure numbers and academic-style captions
               - Reference all figures in the text using Figure numbers
               - Format as: ![Figure X: Description](path/to/image)
               - Include source attribution where appropriate
            
            Use advanced Markdown features throughout. Make the document exceptionally visually appealing
            and adhere to academic publishing standards.
            """),
            HumanMessage(content=format_prompt)
        ])
        
        enhanced_content = self._enhance_image_references(response.content, image_references)
        
        return {
            "title": title,
            "formatted_content": enhanced_content,
            "raw_content": raw_tutorial,
            "image_references": image_references,
            "table_references": table_references,
            "metadata": {
                "date": current_date,
                "citation": self._generate_citation(title, current_date)
            }
        }
    
    def _extract_image_context(self, text: str) -> str:
        patterns = [
            r"(?i)figure\s+\d+[^\n]+",
            r"(?i)image\s+\d+[^\n]+",
            r"(?i)diagram\s+\d+[^\n]+",
            r"(?i)illustration\s+\d+[^\n]+",
            r"(?i)graph\s+\d+[^\n]+",
            r"(?i)chart\s+\d+[^\n]+",
            r"(?i)as shown in[^\n]+",
            r"(?i)depicted in[^\n]+"
        ]
        
        context_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            context_matches.extend(matches)
        
        if context_matches:
            first_match = context_matches[0]
            match_position = text.find(first_match)
            
            paragraph_start = text.rfind("\n\n", 0, match_position)
            if paragraph_start == -1:
                paragraph_start = 0
            else:
                paragraph_start += 2
            
            paragraph_end = text.find("\n\n", match_position)
            if paragraph_end == -1:
                paragraph_end = len(text)
            
            return text[paragraph_start:paragraph_end].strip()
        
        return ""
    
    def _extract_table_context(self, text: str) -> str:
        patterns = [
            r"(?i)table\s+\d+[^\n]+",
            r"(?i)as shown in table[^\n]+",
            r"(?i)listed in table[^\n]+",
            r"(?i)summarized in table[^\n]+",
            r"(?i)the following table[^\n]+"
        ]
        
        context_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            context_matches.extend(matches)
        
        if context_matches:
            first_match = context_matches[0]
            match_position = text.find(first_match)
            
            paragraph_start = text.rfind("\n\n", 0, match_position)
            if paragraph_start == -1:
                paragraph_start = 0
            else:
                paragraph_start += 2
            
            paragraph_end = text.find("\n\n", match_position)
            if paragraph_end == -1:
                paragraph_end = len(text)
            
            return text[paragraph_start:paragraph_end].strip()
        
        return ""
    
    def _generate_image_description(self, basic_description: str, context: str, path: str) -> str:
        if not context:
            return basic_description
        
        image_type_patterns = [
            (r"(?i)graph\s+of\s+([^\n.]+)", "Graph showing"),
            (r"(?i)chart\s+of\s+([^\n.]+)", "Chart depicting"),
            (r"(?i)diagram\s+of\s+([^\n.]+)", "Diagram illustrating"),
            (r"(?i)flow\s*chart\s+of\s+([^\n.]+)", "Flowchart showing"),
            (r"(?i)architecture\s+of\s+([^\n.]+)", "Architecture diagram of"),
            (r"(?i)illustration\s+of\s+([^\n.]+)", "Illustration of"),
            (r"(?i)screenshot\s+of\s+([^\n.]+)", "Screenshot showing")
        ]
        
        for pattern, prefix in image_type_patterns:
            match = re.search(pattern, context)
            if match:
                return f"{prefix} {match.group(1)}"
        
        file_name = path.split("/")[-1] if "/" in path else path.split("\\")[-1] if "\\" in path else path
        return f"{basic_description} - {file_name}"
    
    def _generate_table_description(self, headers: List[str], basic_description: str, context: str) -> str:
        if not headers:
            return basic_description
        
        about_table_patterns = [
            r"(?i)table\s+\w+\s+shows\s+([^\n.]+)",
            r"(?i)table\s+\w+\s+presents\s+([^\n.]+)",
            r"(?i)table\s+\w+\s+summarizes\s+([^\n.]+)",
            r"(?i)table\s+\w+\s+lists\s+([^\n.]+)",
            r"(?i)table\s+\w+\s+contains\s+([^\n.]+)",
            r"(?i)table\s+\w+\s+provides\s+([^\n.]+)"
        ]
        
        for pattern in about_table_patterns:
            match = re.search(pattern, context)
            if match:
                return f"Table showing {match.group(1)}"
        
        header_str = ", ".join(headers)
        return f"Table containing data on {header_str}"
    
    def _get_current_date(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%B %d, %Y")
    
    def _generate_citation(self, title: str, date: str) -> str:
        return f"Author, \"*{title}*,\" {date}."
    
    def _generate_front_matter(self, title: str, written_sections: Dict[str, Any], date: str) -> str:
        learning_objectives = []
        for key, value in written_sections.items():
            if key == "learning_objectives" and isinstance(value, list):
                learning_objectives = value
        
        front_matter = f"""
---
title: "{title}"
date: {date}
author: "Tutorial Generator"
---

"""
        
        introduction = written_sections.get("sections", {}).get("introduction", "")
        if introduction:
            abstract = introduction.split("\n\n")[0] if "\n\n" in introduction else introduction
            front_matter += f"""
## Abstract

{abstract}
"""
        
        if learning_objectives:
            front_matter += """
## Learning Objectives

"""
            for idx, objective in enumerate(learning_objectives):
                front_matter += f"{idx+1}. {objective}\n"
        
        return front_matter
    
    def _enhance_image_references(self, content: str, image_references: List[Dict[str, Any]]) -> str:
        import re
        enhanced_content = content
        
        for img in image_references:
            img_path = re.escape(img['path'])
            simple_pattern = r'!\[.*?\]\(' + img_path + r'\)'
            
            academic_reference = f"![Figure {img['index']}: {img['description']}]({img['path']})"
            academic_caption = f"\n\n*Figure {img['index']}: {img['description']}*"
            
            matches = list(re.finditer(simple_pattern, enhanced_content))
            
            offset = 0
            for match in matches:
                start, end = match.span()
                start += offset
                end += offset
                
                replacement = academic_reference + academic_caption
                enhanced_content = enhanced_content[:start] + replacement + enhanced_content[end:]
                
                offset += len(replacement) - (end - start)
        
        return enhanced_content


class SimpleTutorialImprover:
    def __init__(self, llm_model: str):
        self.llm_model = llm_model
    
    def improve(self, formatted_tutorial: Dict[str, Any]) -> Dict[str, Any]:
        from langchain_ollama import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage
        import json
        import re
        
        llm = ChatOllama(model=self.llm_model)
        
        title = formatted_tutorial.get("title", "Tutorial")
        content = formatted_tutorial.get("formatted_content", "")
        raw_content = formatted_tutorial.get("raw_content", "")
        
        table_references = formatted_tutorial.get("table_references", [])
        image_references = formatted_tutorial.get("image_references", [])
        metadata = formatted_tutorial.get("metadata", {})
        
        already_used_tables = self._identify_used_tables(content)
        already_used_images = self._identify_used_images(content)
        
        table_info = self._generate_table_info(table_references, already_used_tables)
        
        image_info = self._generate_image_info(image_references, already_used_images)
        
        sections = self._identify_tutorial_sections(content)
        
        content_chunks = self._chunk_content(content, max_chunk_size=8000)
        
        all_improvements = []
        
        for chunk_idx, chunk in enumerate(content_chunks):
            analysis_prompt = f"""
            Analyze this part ({chunk_idx+1}/{len(content_chunks)}) of a tutorial and identify ways to improve it:
            
            TUTORIAL TITLE: {title}
            
            CONTENT CHUNK {chunk_idx+1}/{len(content_chunks)}:
            {chunk}
            
            SECTIONS IN THIS CHUNK:
            {json.dumps([s for s in sections if s['title'] in chunk], indent=2)}
            
            AVAILABLE TABLES NOT YET USED IN TUTORIAL:
            {table_info}
            
            AVAILABLE IMAGES NOT YET USED IN TUTORIAL:
            {image_info}
            
            Please identify specific improvements that would make this tutorial more effective, engaging, and academically rigorous.
            Focus especially on:
            1. Finding optimal places to integrate tables and images that are not yet used
            2. Enhancing explanations around existing visual elements
            3. Improving academic quality and depth
            4. Adding explanatory diagrams where concepts could benefit from visualization
            5. Restructuring content for better flow and comprehension
            """
            
            analysis_response = llm.invoke([
                SystemMessage(content="""
                You are an expert academic content reviewer with specialization in educational materials and visual learning.
                Analyze the tutorial chunk and identify specific, actionable improvements in these categories:
                
                1. VISUAL INTEGRATION:
                - Identify optimal places to integrate unused tables and images
                - Suggest ways to better explain existing visual elements
                - Recommend where custom diagrams could enhance understanding
                
                2. ACADEMIC QUALITY:
                - Identify areas where academic rigor could be improved
                - Suggest additional citations or academic context
                - Recommend theoretical frameworks that could be referenced
                
                3. STRUCTURAL IMPROVEMENTS:
                - Identify logical gaps or flow issues
                - Suggest better transitions between concepts
                - Recommend reorganization where beneficial
                
                4. PEDAGOGICAL ENHANCEMENTS:
                - Identify opportunities for improved learning scaffolding
                - Suggest additional examples or case studies
                - Recommend interactive elements or reflection prompts
                
                5. ACCESSIBILITY AND ENGAGEMENT:
                - Identify areas where clarity could be improved
                - Suggest ways to make complex concepts more accessible
                - Recommend engagement techniques for different learning styles
                
                For each suggested improvement, provide:
                - A clear description of the issue
                - A specific, actionable recommendation for addressing it
                - The precise section/location where the improvement should be made
                - Reasoning for why this improvement would enhance learning outcomes
                
                Format your response as a JSON object with an array of improvement objects:
                {
                "improvements": [
                    {
                    "category": "Category name",
                    "issue": "Description of the issue",
                    "recommendation": "Specific recommendation",
                    "location": "Section or location in the tutorial",
                    "reasoning": "Why this improvement matters for learning",
                    "priority": "High/Medium/Low"
                    },
                    ...
                ]
                }
                """),
                HumanMessage(content=analysis_prompt)
            ])
            
            analysis_text = analysis_response.content
            
            try:
                improvements_data = json.loads(analysis_text)
                chunk_improvements = improvements_data.get("improvements", [])
            except json.JSONDecodeError:
                json_match = re.search(r'{.*}', analysis_text, re.DOTALL)
                if json_match:
                    try:
                        extracted_json = json_match.group(0)
                        improvements_data = json.loads(extracted_json)
                        chunk_improvements = improvements_data.get("improvements", [])
                    except json.JSONDecodeError:
                        chunk_improvements = [
                            {
                                "category": "General",
                                "issue": f"Could not parse improvements JSON for chunk {chunk_idx+1}",
                                "recommendation": "Manual review needed",
                                "location": f"Chunk {chunk_idx+1}",
                                "reasoning": "Technical error in analysis",
                                "priority": "High"
                            }
                        ]
                else:
                    chunk_improvements = [
                        {
                            "category": "General",
                            "issue": f"Could not extract improvements data for chunk {chunk_idx+1}",
                            "recommendation": "Manual review needed",
                            "location": f"Chunk {chunk_idx+1}",
                            "reasoning": "Technical error in analysis",
                            "priority": "High"
                        }
                    ]
            
            for improvement in chunk_improvements:
                improvement["chunk_idx"] = chunk_idx
            
            all_improvements.extend(chunk_improvements)
        
        unique_improvements = []
        seen_issues = set()
        for improvement in all_improvements:
            key = (improvement.get("category", ""), improvement.get("issue", ""))
            if key not in seen_issues:
                seen_issues.add(key)
                unique_improvements.append(improvement)
        
        priority_map = {"High": 3, "Medium": 2, "Low": 1}
        unique_improvements.sort(key=lambda x: priority_map.get(x.get("priority", "Low"), 0), reverse=True)
        
        visual_asset_details = self._prepare_visual_assets(table_references, image_references, unique_improvements)
        
        visual_placement_plan = self._generate_visual_placement_plan(content, unique_improvements, visual_asset_details)
        
        improved_chunks = []
        
        for chunk_idx, chunk in enumerate(content_chunks):
            chunk_specific_improvements = [imp for imp in unique_improvements if imp.get("chunk_idx") == chunk_idx]
            
            if not chunk_specific_improvements:
                chunk_specific_improvements = [imp for imp in unique_improvements if "chunk_idx" not in imp]
            
            chunk_specific_improvements = chunk_specific_improvements[:5]
            
            implementation_prompt = f"""
            Please improve this part ({chunk_idx+1}/{len(content_chunks)}) of an academic tutorial based on expert analysis.
            
            TUTORIAL TITLE: {title}
            
            CHUNK {chunk_idx+1}/{len(content_chunks)}:
            {chunk}
            
            IDENTIFIED IMPROVEMENTS FOR THIS CHUNK:
            {json.dumps(chunk_specific_improvements, indent=2)}
            
            VISUAL PLACEMENT PLAN:
            {json.dumps([p for p in visual_placement_plan if any(s["title"] in chunk for s in sections)], indent=2)}
            
            TABLES AVAILABLE FOR INTEGRATION:
            {table_info}
            
            IMAGES AVAILABLE FOR INTEGRATION:
            {image_info}
            
            Apply these improvements to create an enhanced version of this chunk while maintaining:
            1. Seamless integration with other parts of the tutorial
            2. Academic quality with proper citations and theoretical context
            3. Structural coherence and conceptual flow
            4. Pedagogical effectiveness with better examples and explanations
            
            Return ONLY the improved chunk content with all improvements implemented.
            """
            
            implementation_response = llm.invoke([
                SystemMessage(content="""
                You are an expert academic content developer specializing in educational materials with visual integration.
                Your task is to implement all the suggested improvements to transform this tutorial chunk into an exceptional
                learning resource.
                
                Your implementation MUST:
                
                1. VISUAL ELEMENTS:
                - Integrate tables and images at precisely the right locations to support learning
                - Add proper academic captions and explanations for all visual elements
                - Ensure visuals are properly referenced and explained in the surrounding text
                
                2. ACADEMIC ENHANCEMENTS:
                - Add proper citations where recommended
                - Integrate theoretical frameworks where suggested
                - Enhance academic rigor while maintaining accessibility
                - Ensure terminology is precise and consistently used
                
                3. STRUCTURAL IMPROVEMENTS:
                - Implement all suggested reorganizations and transitions
                - Ensure logical flow between sections and concepts
                - Add clear signposting and navigation cues
                - Maintain consistent heading hierarchy
                
                4. MARKDOWN FORMATTING:
                - Use proper Markdown for all elements (tables, figures, code blocks, etc.)
                - Ensure all formatting is consistent throughout
                
                5. PRESERVE CHUNK INTEGRITY:
                - Make sure this chunk will seamlessly connect with other chunks
                - Preserve any necessary context from the chunk
                - Ensure headings maintain proper hierarchy across the document
                
                Return ONLY the improved chunk content with all improvements implemented.
                """),
                HumanMessage(content=implementation_prompt)
            ])
            
            improved_chunks.append(implementation_response.content)
        
        improved_content = "\n\n".join(improved_chunks)
        
        final_content = self._perform_final_refinement(improved_content, visual_asset_details)
        
        return {
            "title": title,
            "original_content": content,
            "improved_content": final_content,
            "improvements": unique_improvements,
            "visual_placement_plan": visual_placement_plan,
            "metadata": metadata
        }
    
    def _identify_tutorial_sections(self, content: str) -> List[Dict[str, Any]]:
        sections = []
        
        heading_pattern = r'#{1,3}\s+(.+?)(?=\n)'
        matches = re.finditer(heading_pattern, content)
        
        current_position = 0
        for match in matches:
            heading = match.group(1).strip()
            heading_level = match.group().count('#')
            start_pos = match.start()
            
            if start_pos > current_position:
                section_content = content[current_position:start_pos].strip()
                if section_content and len(sections) > 0:
                    sections[-1]["content"] = section_content
                    sections[-1]["end_position"] = start_pos
            
            sections.append({
                "title": heading,
                "level": heading_level,
                "start_position": start_pos,
                "content": "",
                "end_position": len(content)
            })
            current_position = match.end()
        
        if sections and current_position < len(content):
            sections[-1]["content"] = content[current_position:].strip()
        
        return sections
    
    def _identify_used_tables(self, content: str) -> List[int]:
        used_tables = []
        
        table_pattern = r'Table\s+(\d+)[:\.]'
        matches = re.finditer(table_pattern, content, re.IGNORECASE)
        
        for match in matches:
            try:
                table_num = int(match.group(1))
                if table_num not in used_tables:
                    used_tables.append(table_num)
            except ValueError:
                continue
        
        return used_tables
    
    def _identify_used_images(self, content: str) -> List[int]:
        used_images = []
        
        figure_pattern = r'Figure\s+(\d+)[:\.]'
        matches = re.finditer(figure_pattern, content, re.IGNORECASE)
        
        for match in matches:
            try:
                figure_num = int(match.group(1))
                if figure_num not in used_images:
                    used_images.append(figure_num)
            except ValueError:
                continue
        
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        image_matches = re.finditer(image_pattern, content)
        for match in image_matches:
            alt_text = match.group(1)
            fig_match = re.search(r'Figure\s+(\d+)', alt_text, re.IGNORECASE)
            if fig_match:
                try:
                    figure_num = int(fig_match.group(1))
                    if figure_num not in used_images:
                        used_images.append(figure_num)
                except ValueError:
                    continue
        
        return used_images
    
    def _generate_table_info(self, table_references: List[Dict[str, Any]], used_tables: List[int]) -> str:
        if not table_references:
            return "No tables available."
        
        table_info = ""
        for table in table_references:
            table_idx = table.get("index", 0)
            
            if table_idx in used_tables:
                continue
                
            description = table.get("description", "No description available")
            headers = table.get("headers", [])
            rows = table.get("rows", 0)
            columns = table.get("columns", 0)
            context = table.get("context", "")
            
            table_info += f"TABLE {table_idx}:\n"
            table_info += f"- Description: {description}\n"
            table_info += f"- Structure: {rows} rows x {columns} columns\n"
            table_info += f"- Headers: {', '.join(headers)}\n"
            
            if table.get("preview") and len(table.get("preview", [])) > 0:
                table_info += "- Sample data:\n"
                for row_idx, row in enumerate(table.get("preview", [])[:2]):
                    table_info += f"  Row {row_idx+1}: {row}\n"
            
            if context:
                table_info += f"- Context: {context}\n"
            
            table_info += "\n"
        
        return table_info
    
    def _generate_image_info(self, image_references: List[Dict[str, Any]], used_images: List[int]) -> str:
        if not image_references:
            return "No images available."
        
        image_info = ""
        for image in image_references:
            image_idx = image.get("index", 0)
            
            if image_idx in used_images:
                continue
                
            description = image.get("description", "No description available")
            path = image.get("path", "")
            context = image.get("context", "")
            
            image_info += f"IMAGE {image_idx}:\n"
            image_info += f"- Description: {description}\n"
            image_info += f"- Path: {path}\n"
            
            if context:
                image_info += f"- Context: {context}\n"
            
            image_info += "\n"
        
        return image_info
    
    def _prepare_visual_assets(
        self, 
        table_references: List[Dict[str, Any]], 
        image_references: List[Dict[str, Any]], 
        improvements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        mentioned_tables = set()
        mentioned_images = set()
        
        for improvement in improvements:
            recommendation = improvement.get("recommendation", "")
            table_mentions = re.findall(r'Table\s+(\d+)', recommendation, re.IGNORECASE)
            for mention in table_mentions:
                try:
                    table_idx = int(mention)
                    mentioned_tables.add(table_idx)
                except ValueError:
                    continue
            
            image_mentions = re.findall(r'(Figure|Image)\s+(\d+)', recommendation, re.IGNORECASE)
            for _, mention in image_mentions:
                try:
                    image_idx = int(mention)
                    mentioned_images.add(image_idx)
                except ValueError:
                    continue
        
        detailed_tables = {}
        for table in table_references:
            table_idx = table.get("index", 0)
            if table_idx in mentioned_tables:
                markdown_table = self._create_markdown_table(
                    headers=table.get("headers", []),
                    preview_data=table.get("preview", [])
                )
                
                detailed_tables[table_idx] = {
                    "description": table.get("description", ""),
                    "markdown_table": markdown_table,
                    "context": table.get("context", ""),
                    "reference_text": f"Table {table_idx}: {table.get('description', '')}"
                }
        
        detailed_images = {}
        for image in image_references:
            image_idx = image.get("index", 0)
            if image_idx in mentioned_images:
                detailed_images[image_idx] = {
                    "description": image.get("description", ""),
                    "path": image.get("path", ""),
                    "context": image.get("context", ""),
                    "markdown_image": f"![Figure {image_idx}: {image.get('description', '')}]({image.get('path', '')})\n\n*Figure {image_idx}: {image.get('description', '')}*",
                    "reference_text": f"Figure {image_idx}"
                }
        
        return {
            "tables": detailed_tables,
            "images": detailed_images
        }
    
    def _create_markdown_table(self, headers: List[str], preview_data: List[Dict[str, Any]], max_rows: int = 5) -> str:
        if not headers or not preview_data:
            return "*Table data not available*"
        
        markdown_table = "| " + " | ".join(headers) + " |\n"
        markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row_idx, row in enumerate(preview_data[:max_rows]):
            row_values = []
            for header in headers:
                row_values.append(str(row.get(header, "")))
            markdown_table += "| " + " | ".join(row_values) + " |\n"
        
        return markdown_table
    
    def _generate_visual_placement_plan(
        self, 
        content: str, 
        improvements: List[Dict[str, Any]], 
        visual_asset_details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        placement_plan = []
        
        sections = self._identify_tutorial_sections(content)
        
        for improvement in improvements:
            category = improvement.get("category", "").lower()
            recommendation = improvement.get("recommendation", "").lower()
            location = improvement.get("location", "")
            
            involves_table = "table" in category or "table" in recommendation
            involves_image = any(term in category or term in recommendation for term in ["image", "figure", "visual"])
            
            if involves_table or involves_image:
                target_section = None
                for section in sections:
                    if location.lower() in section["title"].lower():
                        target_section = section
                        break
                
                if not target_section and sections:
                    from difflib import SequenceMatcher
                    
                    best_match = None
                    best_ratio = 0
                    
                    for section in sections:
                        ratio = SequenceMatcher(None, location.lower(), section["title"].lower()).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match = section
                    
                    if best_ratio > 0.3:
                        target_section = best_match
                
                asset_numbers = []
                if involves_table:
                    table_mentions = re.findall(r'Table\s+(\d+)', improvement.get("recommendation", ""), re.IGNORECASE)
                    asset_numbers = table_mentions
                elif involves_image:
                    image_mentions = re.findall(r'(Figure|Image)\s+(\d+)', improvement.get("recommendation", ""), re.IGNORECASE)
                    asset_numbers = [match[1] for match in image_mentions]
                
                for number_str in asset_numbers:
                    try:
                        asset_number = int(number_str)
                        
                        placement_entry = {
                            "asset_type": "table" if involves_table else "image",
                            "asset_number": asset_number,
                            "target_location": location,
                            "target_section": target_section["title"] if target_section else "Best suitable section",
                            "recommendation": improvement.get("recommendation", ""),
                            "reasoning": improvement.get("reasoning", "")
                        }
                        
                        placement_plan.append(placement_entry)
                    except ValueError:
                        continue
        
        return placement_plan
    
    def _perform_final_refinement(self, content: str, visual_asset_details: Dict[str, Any]) -> str:
        for table_idx, table_info in visual_asset_details["tables"].items():
            table_marker = f"Table {table_idx}:"
            
            if table_marker in content and table_info["markdown_table"] not in content:
                table_mention_pattern = f"(Table {table_idx}:.*?)(\n\n|\n#{1,3}\s)"
                match = re.search(table_mention_pattern, content, re.DOTALL)
                
                if match:
                    replacement = f"**{table_marker} {table_info['description']}**\n\n{table_info['markdown_table']}\n\n"
                    content = content[:match.start()] + replacement + content[match.end()-len(match.group(2)):]
        
        for image_idx, image_info in visual_asset_details["images"].items():
            image_marker = f"Figure {image_idx}:"
            
            if image_marker in content and image_info["path"] not in content:
                image_mention_pattern = f"(Figure {image_idx}:.*?)(\n\n|\n#{1,3}\s)"
                match = re.search(image_mention_pattern, content, re.DOTALL)
                
                if match:
                    replacement = f"{image_info['markdown_image']}\n\n"
                    content = content[:match.start()] + replacement + content[match.end()-len(match.group(2)):]
        
        content = re.sub(r'(#{1,6})([^ ])', r'\1 \2', content)
        
        content = re.sub(r'\n{3,}', r'\n\n', content)
        
        open_code_blocks = content.count("```") % 2
        if open_code_blocks:
            content += "\n```\n"
        
        content = self._enhance_visual_references(content, visual_asset_details)
        
        content = self._enhance_academic_citations(content)
        
        return content
    
    def _enhance_visual_references(self, content: str, visual_asset_details: Dict[str, Any]) -> str:
        enhanced_content = content
        
        for table_idx, table_info in visual_asset_details["tables"].items():
            simple_ref_pattern = f"(table {table_idx})([^:])"
            enhanced_content = re.sub(
                simple_ref_pattern, 
                f"Table {table_idx} ('{table_info['description']}')\g<2>", 
                enhanced_content, 
                flags=re.IGNORECASE
            )
            
            headers_joined = "|".join(re.escape(h) for h in table_info.get("headers", []) if h)
            if headers_joined:
                header_pattern = f"({headers_joined})"
                matches = list(re.finditer(header_pattern, enhanced_content, re.IGNORECASE))
                
                if matches and f"Table {table_idx}" not in enhanced_content[:matches[0].start()]:
                    first_match = matches[0]
                    enhanced_content = (
                        enhanced_content[:first_match.start()] + 
                        f"As shown in Table {table_idx}, " + 
                        enhanced_content[first_match.start():]
                    )
        
        for image_idx, image_info in visual_asset_details["images"].items():
            simple_ref_pattern = f"(figure {image_idx})([^:])"
            enhanced_content = re.sub(
                simple_ref_pattern, 
                f"Figure {image_idx} ('{image_info['description']}')\g<2>", 
                enhanced_content, 
                flags=re.IGNORECASE
            )
            
            if image_info["path"] in enhanced_content and f"Figure {image_idx}" not in enhanced_content:
                img_pos = enhanced_content.find(image_info["path"])
                paragraph_end = enhanced_content.rfind("\n\n", 0, img_pos)
                
                if paragraph_end > 0:
                    enhanced_content = (
                        enhanced_content[:paragraph_end] + 
                        f"\n\nAs illustrated in Figure {image_idx}, {image_info['description']}." +
                        enhanced_content[paragraph_end:]
                    )
        
        return enhanced_content
    
    def _enhance_academic_citations(self, content: str) -> str:
        enhanced_content = content
        
        has_references = re.search(r'#+\s+(References|Bibliography)', enhanced_content, re.IGNORECASE) is not None
        
        academic_statement_patterns = [
            r'studies (have shown|indicate|demonstrate)',
            r'research (has shown|indicates|demonstrates)',
            r'according to',
            r'it has been (shown|demonstrated|proven)',
            r'(evidence|data) suggests'
        ]
        
        citation_count = 0
        for pattern in academic_statement_patterns:
            matches = list(re.finditer(pattern, enhanced_content, re.IGNORECASE))
            
            for match in matches:
                surrounding_text = enhanced_content[max(0, match.start()-50):min(len(enhanced_content), match.end()+50)]
                if re.search(r'\[\d+\]', surrounding_text):
                    continue
                    
                citation_count += 1
                enhanced_content = (
                    enhanced_content[:match.end()] + 
                    f" [{citation_count}]" + 
                    enhanced_content[match.end():]
                )
        
        if citation_count > 0 and not has_references:
            enhanced_content += "\n\n## References\n\n"
            for i in range(1, citation_count + 1):
                enhanced_content += f"[{i}] Author, A. (Year). *Title of the work*. Publication Source.\n\n"
        
        return enhanced_content
    def _chunk_content(self, content: str, max_chunk_size: int = 9000) -> List[str]:
        sections = re.split(r'(#+\s+.*?\n)', content)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) <= max_chunk_size:
                current_chunk += section
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = section
        
        if current_chunk:
            chunks.append(current_chunk)
        
        if not chunks or any(len(chunk) > max_chunk_size for chunk in chunks):
            chunks = []
            for i in range(0, len(content), max_chunk_size):
                chunks.append(content[i:i+max_chunk_size])
        
        return chunks
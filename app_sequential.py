import streamlit as st

# Set page config - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="ReAct Tutorial Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import requests
import json
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from embedding.text_embedding import get_available_text_embedding_models
from embedding.image_embedding import get_available_image_embedding_models
from embedding.tables_embedding import get_available_table_embedding_models
from rag.rag_pipeline import RAGPipeline
from agents.sequential_agent import SequentialReActAgent

# Initialize session state variables if they don't exist
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "tutorial_agent" not in st.session_state:
    st.session_state.tutorial_agent = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "available_models" not in st.session_state:
    st.session_state.available_models = {
        "llm": [],
        "text_embedding": [],
        "image_embedding": [], 
        "table_embedding": []
    }
if "default_models" not in st.session_state:
    st.session_state.default_models = {
        "llm": "",
        "text_embedding": "",
        "image_embedding": "",
        "table_embedding": ""
    }
if "tutorial_result" not in st.session_state:
    st.session_state.tutorial_result = None
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = None
if "tutorial_plan" not in st.session_state:
    st.session_state.tutorial_plan = None
if "generation_logs" not in st.session_state:
    st.session_state.generation_logs = []

# Function to get available models from Ollama
def get_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            all_models = response.json().get("models", [])
            
            # Get all model names
            all_model_names = [model["name"] for model in all_models]
            
            # Add these models to all categories to ensure users can select any model
            st.session_state.available_models = {
                "llm": all_model_names,
                "text_embedding": all_model_names,
                "image_embedding": all_model_names,
                "table_embedding": all_model_names
            }
            
            # Try to intelligently suggest defaults based on model names
            st.session_state.default_models = {
                "llm": next((model for model in all_model_names if "llama" in model.lower()), all_model_names[0] if all_model_names else ""),
                "text_embedding": next((model for model in all_model_names if any(name in model.lower() for name in ["embed", "nomic", "e5"])), all_model_names[0] if all_model_names else ""),
                "image_embedding": next((model for model in all_model_names if any(name in model.lower() for name in ["llava", "vision", "clip"])), all_model_names[0] if all_model_names else ""),
                "table_embedding": next((model for model in all_model_names if "embed" in model.lower()), all_model_names[0] if all_model_names else "")
            }
            
            st.session_state.models_loaded = True
            return True
        else:
            st.error("Error fetching models from Ollama API")
            return False
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        st.info("Make sure Ollama is running on http://localhost:11434")
        return False

# Function to initialize the system with sequential ReAct agent
def initialize_system(text_embedding_model, image_embedding_model, table_embedding_model, llm_model):
    try:
        # Initialize RAG pipeline
        st.session_state.rag_pipeline = RAGPipeline(
            collection_name="tutorial_collection",
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model,
            table_embedding_model=table_embedding_model,
            llm_model=llm_model,
            persist_dir="./chroma_db"
        )
        
        # Initialize tutorial agent with sequential ReAct agent
        st.session_state.tutorial_agent = SequentialReActAgent(
            llm_model=llm_model,
            collection_name="tutorial_collection",
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model
        )
        
        # Clear previous logs
        st.session_state.generation_logs = []
        
        return True
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return False

# Function to generate tutorial with progress updates
def generate_tutorial(pdf_path, goal):
    try:
        # Clear previous logs
        st.session_state.generation_logs = []
        
        # Create progress bar and status areas
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_area = st.empty()
        
        # Capture print statements for logging
        import sys
        from io import StringIO
        
        class LogCapture:
            def __init__(self, status_text, log_area):
                self.logs = []
                self.status_text = status_text
                self.log_area = log_area
                
            def write(self, text):
                if text.strip():  # Only process non-empty lines
                    self.logs.append(text.strip())
                    # Update the session state logs
                    st.session_state.generation_logs = self.logs
                    # Update the log area with all logs
                    self.log_area.code("\n".join(self.logs), language="")
                    # Update status with the last line
                    self.status_text.text(text.strip())
                
            def flush(self):
                pass
        
        # Redirect stdout to capture logs
        old_stdout = sys.stdout
        log_capture = LogCapture(status_text, log_area)
        sys.stdout = log_capture
        
        try:
            # Use the SequentialReActAgent to generate the tutorial
            result = st.session_state.tutorial_agent.run(pdf_path, goal)
            
            # Update the progress to 100%
            progress_bar.progress(100)
            status_text.text("Tutorial generation complete!")
            
            # Store the result
            st.session_state.tutorial_result = {
                "title": result.get("plan", {}).get("title", "Generated Tutorial") if result.get("plan") else "Generated Tutorial",
                "content": result.get("tutorial", ""),
                "improvements": result.get("improvements", []),
                "plan": result.get("plan", {}),
                "logs": st.session_state.generation_logs.copy()
            }
            
            # Store PDF content and tutorial plan if available
            if "pdf_content" in result:
                st.session_state.pdf_content = result["pdf_content"]
            
            if "plan" in result:
                st.session_state.tutorial_plan = result["plan"]
            
            return st.session_state.tutorial_result
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout
    
    except Exception as e:
        st.error(f"Error generating tutorial: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Main UI
st.title("ReAct Tutorial Generator")
st.write("Convert PDF documents into educational tutorials using ReAct agent architecture")

# Sidebar for model selection
st.sidebar.title("Model Selection")

# Load available models
if not st.session_state.models_loaded:
    if st.sidebar.button("Load Available Models"):
        # Create a placeholder in the sidebar for status messages
        status_placeholder = st.sidebar.empty()
        status_placeholder.info("Loading models from Ollama...")
        success = get_available_models()
        if success:
            status_placeholder.success("Models loaded successfully!")
        else:
            status_placeholder.error("Failed to load models. Check Ollama server.")

# Model selection dropdowns (only show if models are loaded)
if st.session_state.models_loaded:
    st.sidebar.subheader("Select Models")
    
    
    # LLM model selection
    llm_options = st.session_state.available_models["llm"]
    default_llm = st.session_state.default_models["llm"]
    selected_llm = st.sidebar.selectbox(
        "LLM Model",
        options=llm_options,
        index=llm_options.index(default_llm) if default_llm in llm_options else 0,
        help="Language model for text generation"
    )
    
    # Text embedding model selection
    text_embed_options = st.session_state.available_models["text_embedding"]
    default_text_embed = st.session_state.default_models["text_embedding"]
    selected_text_embed = st.sidebar.selectbox(
        "Text Embedding Model",
        options=text_embed_options,
        index=text_embed_options.index(default_text_embed) if default_text_embed in text_embed_options else 0,
        help="Model for text embedding"
    )
    
    # Image embedding model selection
    img_embed_options = st.session_state.available_models["image_embedding"]
    default_img_embed = st.session_state.default_models["image_embedding"]
    selected_img_embed = st.sidebar.selectbox(
        "Image Embedding Model",
        options=img_embed_options,
        index=img_embed_options.index(default_img_embed) if default_img_embed in img_embed_options else 0,
        help="Model for image embedding/understanding"
    )
    
    # Table embedding model selection
    table_embed_options = st.session_state.available_models["table_embedding"]
    default_table_embed = st.session_state.default_models["table_embedding"]
    selected_table_embed = st.sidebar.selectbox(
        "Table Embedding Model",
        options=table_embed_options,
        index=table_embed_options.index(default_table_embed) if default_table_embed in table_embed_options else 0,
        help="Model for table/structured data embedding"
    )
    
    # Initialize button
    if st.sidebar.button("Initialize System with Selected Models"):
        # Create a placeholder in the sidebar for status messages
        init_status = st.sidebar.empty()
        init_status.info("Initializing system...")
        success = initialize_system(
            selected_text_embed,
            selected_img_embed,
            selected_table_embed,
            selected_llm
        )
        if success:
            init_status.success("System initialized successfully!")
        else:
            init_status.error("Failed to initialize system.")

# Main panel with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Generate Tutorial", "View PDF Content", "Tutorial Plan", "Process Logs"])

# Tab 1: Generate Tutorial
with tab1:
    st.header("Generate Tutorial from PDF")
    
    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file
        pdf_dir = Path("uploaded_pdfs")
        pdf_dir.mkdir(exist_ok=True)
        pdf_path = pdf_dir / uploaded_file.name
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"PDF saved to {pdf_path}")
        
        # Tutorial goal input
        tutorial_goal = st.text_area(
            "What is the goal of this tutorial?",
            "Create a comprehensive tutorial that explains the key concepts in the document and provides practical examples."
        )
        
        # Generate button (only enabled if system is initialized)
        if st.session_state.tutorial_agent is not None:
            if st.button("Generate Tutorial"):
                with st.spinner("Generating tutorial... This may take several minutes."):
                    result = generate_tutorial(str(pdf_path), tutorial_goal)
                    if result:
                        st.success("Tutorial generated successfully!")
        else:
            st.warning("Please initialize the system with selected models first (using the sidebar)")
    
    # Display generated tutorial if available
    if st.session_state.tutorial_result:
        st.header(st.session_state.tutorial_result.get("title", "Generated Tutorial"))
        st.markdown(st.session_state.tutorial_result.get("content", ""))
        
        # Download button for tutorial
        tutorial_md = st.session_state.tutorial_result.get("content", "")
        st.download_button(
            label="Download Tutorial as Markdown",
            data=tutorial_md,
            file_name="generated_tutorial.md",
            mime="text/markdown"
        )
        
        # Display improvements
        with st.expander("View Tutorial Improvements"):
            improvements = st.session_state.tutorial_result.get("improvements", [])
            for i, improvement in enumerate(improvements):
                st.subheader(f"Improvement {i+1}: {improvement.get('category', '')}")
                st.write(f"**Issue:** {improvement.get('issue', '')}")
                st.write(f"**Recommendation:** {improvement.get('recommendation', '')}")
                st.write(f"**Location:** {improvement.get('location', '')}")
                st.divider()

# Tab 2: View PDF Content
with tab2:
    st.header("PDF Content")
    
    if st.session_state.pdf_content:
        # Display PDF metadata
        metadata = st.session_state.pdf_content.get("metadata", {})
        st.subheader("Document Metadata")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Title:** {metadata.get('title', 'N/A')}")
            st.write(f"**Author:** {metadata.get('author', 'N/A')}")
            st.write(f"**Subject:** {metadata.get('subject', 'N/A')}")
        with col2:
            st.write(f"**Creation Date:** {metadata.get('creation_date', 'N/A')}")
            st.write(f"**Producer:** {metadata.get('producer', 'N/A')}")
            st.write(f"**Total Pages:** {st.session_state.pdf_content.get('total_pages', 0)}")
        
        # Display page content
        st.subheader("Page Content")
        pages = st.session_state.pdf_content.get("pages", [])
        
        if pages:
            page_selection = st.selectbox(
                "Select page to view",
                options=list(range(1, len(pages) + 1))
            )
            
            # Display selected page
            selected_page = pages[page_selection - 1]
            st.write(f"**Page {selected_page.get('page_num', '')}**")
            
            # Display text with scrollable container
            with st.expander("Page Text", expanded=True):
                st.text_area(
                    "Content",
                    value=selected_page.get("text", ""),
                    height=400,
                    disabled=True
                )
            
            # Display images if any
            page_images = selected_page.get("images", [])
            if page_images:
                st.subheader(f"Images on Page {page_selection}")
                
                # Create columns for images
                img_cols = st.columns(min(3, len(page_images)))
                for i, img in enumerate(page_images):
                    with img_cols[i % len(img_cols)]:
                        try:
                            st.image(
                                img.get("path", ""),
                                caption=f"Image {i+1}",
                                use_container_width=True
                            )
                        except:
                            st.error(f"Could not display image {i+1}")
    else:
        st.info("No PDF content available. Please upload a PDF and generate a tutorial first.")

# Tab 3: Tutorial Plan
with tab3:
    st.header("Tutorial Plan")
    
    if st.session_state.tutorial_plan:
        plan = st.session_state.tutorial_plan
        
        # Display plan header
        st.subheader(plan.get("title", "Tutorial"))
        st.write(f"**Goal:** {plan.get('tutorial_goal', '')}")
        st.write(f"**Target Audience:** {plan.get('target_audience', '')}")
        
        # Display learning objectives
        st.subheader("Learning Objectives")
        for i, objective in enumerate(plan.get("learning_objectives", [])):
            st.write(f"{i+1}. {objective}")
        
        # Display introduction
        st.subheader("Introduction")
        st.write(plan.get("introduction", ""))
        
        # Display sections
        st.subheader("Sections")
        for i, section in enumerate(plan.get("sections", [])):
            with st.expander(f"{i+1}. {section.get('title', '')}"):
                st.write(f"**Summary:** {section.get('content_summary', '')}")
                
                # Key points
                st.write("**Key Points:**")
                for point in section.get("key_points", []):
                    st.write(f"- {point}")
                
                # Subsections
                if section.get("subsections"):
                    st.write("**Subsections:**")
                    for j, subsection in enumerate(section.get("subsections", [])):
                        st.write(f"- {subsection.get('title', '')}")
                
                # Examples
                if section.get("examples"):
                    st.write("**Examples:**")
                    for example in section.get("examples", []):
                        st.write(f"- {example}")
                
                # Exercises
                if section.get("exercises"):
                    st.write("**Exercises:**")
                    for exercise in section.get("exercises", []):
                        st.write(f"- {exercise}")
        
        # Display conclusion
        st.subheader("Conclusion")
        st.write(plan.get("conclusion", ""))
        
        # Display estimated duration
        if "estimated_duration" in plan:
            st.write(f"**Estimated Duration:** {plan.get('estimated_duration', '')}")
    else:
        st.info("No tutorial plan available. Please upload a PDF and generate a tutorial first.")

# Tab 4: Process Logs
with tab4:
    st.header("Generation Process Logs")
    
    if st.session_state.generation_logs:
        # Add options for log display
        log_display_options = st.radio(
            "Log Display Options",
            ["Full Logs", "Thinking Process Only", "Actions Only"],
            horizontal=True
        )
        
        # Filter logs based on selection
        filtered_logs = st.session_state.generation_logs
        
        if log_display_options == "Thinking Process Only":
            filtered_logs = [log for log in st.session_state.generation_logs if "Thought:" in log]
        elif log_display_options == "Actions Only":
            filtered_logs = [log for log in st.session_state.generation_logs if "Executing tool:" in log or "Action:" in log]
        
        # Add a download button for logs
        st.download_button(
            label="Download Logs",
            data="\n".join(st.session_state.generation_logs),
            file_name="tutorial_generation_logs.txt",
            mime="text/plain"
        )
        
        # Display the ReAct agent logs with syntax highlighting for different parts
        log_text = "\n".join(filtered_logs)
        
        # Add visual separators for iterations
        log_text = log_text.replace("----- Iteration", "━━━━━━━━━━━━━━━━━━━━━━━\n----- Iteration")
        
        # Display logs in a code block
        st.code(log_text, language="")
        
        # Display a visual ReAct flow diagram
        with st.expander("ReAct Process Visualization"):
            st.markdown("""```mermaid
            graph LR
                A[Observe] --> B[Think]
                B --> C[Act]
                C --> A
                C --> D[Complete]
                
                style A fill:#f9f,stroke:#333,stroke-width:2px
                style B fill:#bbf,stroke:#333,stroke-width:2px
                style C fill:#bfb,stroke:#333,stroke-width:2px
                style D fill:#ff9,stroke:#333,stroke-width:2px
            ```""")
            
            # Count iterations of each step
            observe_count = sum(1 for log in st.session_state.generation_logs if "Observation:" in log)
            think_count = sum(1 for log in st.session_state.generation_logs if "Thought:" in log)
            act_count = sum(1 for log in st.session_state.generation_logs if "Executing tool:" in log)
            
            # Display statistics
            st.write(f"**Observe Steps:** {observe_count}")
            st.write(f"**Think Steps:** {think_count}")
            st.write(f"**Act Steps:** {act_count}")
            st.write(f"**Total Iterations:** {max(observe_count, think_count, act_count)}")
    else:
        st.info("No generation logs available. Generate a tutorial to see the process logs.")
        
        # Add an example of what the ReAct process looks like
        with st.expander("What to expect in Process Logs"):
            st.markdown("""The ReAct agent follows these steps:
            
            1. **Observe** - Analyzes the current state
            ```
            Observation: Agent initialized. Ready to process the PDF...
            ```
            
            2. **Think** - Reasons about what to do next
            ```
            Thought: I need to first extract the PDF content before I can process it...
            ```
            
            3. **Act** - Executes a tool
            ```
            Executing tool: extract_pdf
            ```
            
            This cycle repeats until the tutorial is complete.
            """)

# Footer
st.divider()
st.caption("ReAct Tutorial Generator | PDF → Tutorial with Reasoning and Acting agents")
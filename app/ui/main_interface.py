"""
Main Streamlit interface for the tutorial generator application.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

import streamlit as st

logger = logging.getLogger(__name__)

def _initialize_session_state():
    """Initialize Streamlit session state."""
    if "tutorial_content" not in st.session_state:
        st.session_state.tutorial_content = {}
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "current_state" not in st.session_state:
        st.session_state.current_state = {}
    
    if "settings" not in st.session_state:
        st.session_state.settings = {}
    
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = True

def _copy_to_clipboard(content):
    """Copy content to clipboard using JavaScript."""
    # Create a JavaScript function to copy text
    js_code = f"""
    <script>
    function copyToClipboard() {{
        const el = document.createElement('textarea');
        el.value = `{content.replace('`', '``')}`;
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
        
        // Show a toast notification
        const toast = document.createElement('div');
        toast.innerHTML = 'Copied to clipboard!';
        toast.style.position = 'fixed';
        toast.style.bottom = '20px';
        toast.style.left = '50%';
        toast.style.transform = 'translateX(-50%)';
        toast.style.backgroundColor = '#4CAF50';
        toast.style.color = 'white';
        toast.style.padding = '16px';
        toast.style.borderRadius = '4px';
        toast.style.zIndex = '9999';
        document.body.appendChild(toast);
        
        // Remove the toast after 3 seconds
        setTimeout(() => {{
            document.body.removeChild(toast);
        }}, 3000);
    }}
    copyToClipboard();
    </script>
    """
    
    # Inject JavaScript
    st.components.v1.html(js_code, height=0)

def _save_tutorial_to_outputs(config, content):
    """Save tutorial content to outputs directory.
    
    Args:
        config: Application configuration
        content: Tutorial content
    """
    try:
        import time
        import os
        
        # Create outputs directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tutorial_{timestamp}.md"
        filepath = os.path.join(config.output_dir, filename)
        
        # Write content to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        st.success(f"Tutorial saved to: {filepath}")
    except Exception as e:
        st.error(f"Error saving tutorial: {str(e)}")

def _create_edit_interface():
    """Create an interface for editing tutorial content."""
    st.subheader(f"Editing: {st.session_state.editing_section}")
    
    # Editable content area
    edited_content = st.text_area(
        "Edit Content",
        value=st.session_state.editing_content,
        height=400
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Changes", use_container_width=True):
            # Update the content in session state
            st.session_state.tutorial_content[st.session_state.editing_section] = edited_content
            # Clear editing state
            st.session_state.editing_section = None
            st.session_state.editing_content = None
            st.success("Changes saved!")
            st.experimental_rerun()
    
    with col2:
        if st.button("Cancel", use_container_width=True):
            # Clear editing state
            st.session_state.editing_section = None
            st.session_state.editing_content = None
            st.experimental_rerun()

def _create_sidebar(config, model_provider):
    """Create sidebar with settings and model selection.
    
    Args:
        config: Application configuration
        model_provider: Provider for accessing models
    """
    with st.sidebar:
        st.title("Settings")
        
        # Fetch available models from Ollama API
        try:
            available_models = model_provider.list_models()
            
            # Group models by type
            llm_models = []
            embedding_models = []
            vision_models = []
            
            for model in available_models:
                model_name = model.get("name", "")
                
                # Check tags or name for model type
                tags = model.get("tags", [])
                if not tags:
                    tags = []
                elif isinstance(tags, str):
                    tags = [tags]
                
                if "embed" in model_name.lower() or any("embed" in tag.lower() for tag in tags):
                    embedding_models.append(model)
                elif "vision" in model_name.lower() or any("vision" in tag.lower() for tag in tags):
                    vision_models.append(model)
                else:
                    llm_models.append(model)
            
            # If no models found in any category, show warning
            if not llm_models and not embedding_models and not vision_models:
                st.warning("No models found. Make sure Ollama is running and models are installed.")
                st.markdown("""
                Install models with:
                ```
                ollama pull llama3
                ollama pull nomic-embed-text
                ```
                """)
            
            # LLM Model Selection
            st.header("Language Models")
            if llm_models:
                llm_options = [m["name"] for m in llm_models]
                default_llm = st.session_state.get("selected_llm", llm_options[0] if llm_options else None)
                
                selected_llm = st.selectbox(
                    "Select LLM Model",
                    options=llm_options,
                    index=llm_options.index(default_llm) if default_llm in llm_options else 0,
                    help="Language model for text generation"
                )
                
                if selected_llm:
                    st.session_state.selected_llm = selected_llm
            else:
                st.info("No language models found. Please install some models using Ollama CLI.")
            
            # Embedding Model Selection
            st.header("Embedding Models")
            if embedding_models:
                embed_options = [m["name"] for m in embedding_models]
                default_embed = st.session_state.get("selected_embedding", embed_options[0] if embed_options else None)
                
                selected_embed = st.selectbox(
                    "Select Embedding Model",
                    options=embed_options,
                    index=embed_options.index(default_embed) if default_embed in embed_options else 0,
                    help="Model for creating vector embeddings"
                )
                
                if selected_embed:
                    st.session_state.selected_embedding = selected_embed
            elif llm_models:
                # If no dedicated embedding models, use LLMs as fallback
                embed_options = [m["name"] for m in llm_models]
                default_embed = st.session_state.get("selected_embedding", embed_options[0] if embed_options else None)
                
                st.info("No dedicated embedding models found. Using language models instead.")
                selected_embed = st.selectbox(
                    "Select Embedding Model",
                    options=embed_options,
                    index=embed_options.index(default_embed) if default_embed in embed_options else 0,
                    help="Model for creating vector embeddings"
                )
                
                if selected_embed:
                    st.session_state.selected_embedding = selected_embed
            else:
                st.info("No embedding models found. Please install some models using Ollama CLI.")
            
            # Vision Model Selection
            if vision_models:
                st.header("Vision Models")
                vision_options = [m["name"] for m in vision_models]
                default_vision = st.session_state.get("selected_vision", vision_options[0] if vision_options else None)
                
                selected_vision = st.selectbox(
                    "Select Vision Model",
                    options=vision_options,
                    index=vision_options.index(default_vision) if default_vision in vision_options else 0,
                    help="Model for processing images"
                )
                
                if selected_vision:
                    st.session_state.selected_vision = selected_vision
        
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Make sure Ollama is running and accessible.")
        
        # Web search toggle
        st.header("Web Search")
        enable_web_search = st.checkbox(
            "Enable Web Search", 
            value=config.enable_web_search,
            help="Enable web search to supplement document content"
        )
        
        # Debug mode toggle
        st.header("Debug")
        debug_mode = st.checkbox(
            "Show Debug Panel", 
            value=st.session_state.debug_mode,
            help="Show agent debugging information"
        )
        
        # Model settings
        with st.expander("Model Settings", expanded=False):
            # Temperature
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("temperature", 0.7),
                step=0.1,
                help="Higher values make output more random, lower values more deterministic"
            )
            st.session_state.temperature = temperature
            
            # Max tokens
            max_tokens = st.slider(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=st.session_state.get("max_tokens", 2000),
                step=100,
                help="Maximum number of tokens to generate"
            )
            st.session_state.max_tokens = max_tokens
        
        # Apply settings
        if st.button("Apply Settings"):
            st.session_state.settings = {
                "llm_model": st.session_state.get("selected_llm", config.default_llm_model),
                "embedding_model": st.session_state.get("selected_embedding", config.default_embedding_model),
                "vision_model": st.session_state.get("selected_vision"),
                "enable_web_search": enable_web_search,
                "debug_mode": debug_mode,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Update config
            config.enable_web_search = enable_web_search
            config.agent_temperature = temperature
            config.agent_max_tokens = max_tokens
            st.session_state.debug_mode = debug_mode
            
            st.success("Settings applied!")
        
        # About section
        st.header("About")
        st.markdown("""
        **AI Tutorial Generator**
        
        This application generates comprehensive tutorials from PDF documents using:
        
        - 🧠 Agent-based architecture
        - 🔍 RAG for content retrieval
        - 🖼️ Multimodal content processing
        - 🤖 Local AI models via Ollama
        
        Upload a PDF and specify what tutorial you want to create!
        """)

def _create_help_section():
    """Create help section with instructions and FAQs."""
    st.header("Help & Instructions")
    
    # Basic usage
    st.subheader("Basic Usage")
    st.markdown("""
    1. **Upload a PDF Document**: In the "Create Tutorial" tab, upload a PDF document that contains the information you want to use for your tutorial.
    
    2. **Specify Tutorial Topic**: Enter what kind of tutorial you want to generate. Be specific about your needs. For example: "Create a beginner's tutorial on machine learning using this textbook" or "Generate a step-by-step guide for setting up a web server based on this manual."
    
    3. **Configure Models**: In the sidebar, select which AI models to use. If you have multiple models installed via Ollama, you can choose the most appropriate ones.
    
    4. **Generate Tutorial**: Click the "Generate Tutorial" button and wait for the system to process your document and create the tutorial.
    
    5. **View and Export**: Once generated, switch to the "Generated Output" tab to view your tutorial. You can download it as a markdown file or copy it to your clipboard.
    """)
    
    # FAQs
    st.subheader("Frequently Asked Questions")
    
    with st.expander("What kinds of PDFs work best?"):
        st.markdown("""
        The best results come from PDFs that are:
        
        - Text-based (not scanned images)
        - Well-structured with clear headings and sections
        - Contain relevant content for your tutorial topic
        - Include diagrams, tables, or images that illustrate concepts
        
        Technical documentation, textbooks, guides, and manuals work particularly well.
        """)
    
    with st.expander("How does the agent system work?"):
        st.markdown("""
        The system uses three specialized AI agents that work together:
        
        1. **Planning Agent**: Analyzes your document and query to create an outline for the tutorial
        2. **Research Agent**: Gathers relevant information for each section from the document and (optionally) the web
        3. **Content Agent**: Generates the actual tutorial content, including text explanations and incorporating relevant images and tables
        
        Each agent follows a "Reasoning, Action, Observation" cycle to work through its tasks intelligently.
        """)
    
    with st.expander("Why use local models instead of API services?"):
        st.markdown("""
        This system uses Ollama to run models locally for several advantages:
        
        - **Privacy**: Your documents stay on your machine
        - **No API costs**: No usage fees or token limits
        - **Customization**: Use any compatible model
        - **Offline capability**: Works without internet connection (except for web search feature)
        
        The tradeoff is that you need sufficient computational resources to run the models.
        """)
    
    with st.expander("How can I improve the quality of generated tutorials?"):
        st.markdown("""
        To get better results:
        
        - Use more specific queries that clearly describe your tutorial goals
        - Provide PDFs with comprehensive, relevant content
        - Enable web search for supplementary information
        - Try different models for different types of content
        - Adjust the temperature setting (lower for more focused content, higher for more creative)
        - Edit the generated content to refine or expand sections
        """)
    
    with st.expander("What if I get an error?"):
        st.markdown("""
        Common issues and solutions:
        
        - **"No models found"**: Make sure Ollama is running and you have models installed
        - **Processing fails**: Your PDF might be too large or complex; try a smaller document
        - **Memory errors**: Close other applications to free up system resources
        - **Slow generation**: Large documents take time to process; be patient or use a smaller document
        
        If errors persist, check the console output for more detailed error messages.
        """)
    
    # System requirements
    st.subheader("System Requirements")
    st.markdown("""
    - Python 3.9 or higher
    - Ollama installed and running
    - Sufficient RAM for running language models (8GB minimum, 16GB+ recommended)
    - Storage space for models and generated content
    """)
    
    # Tips and tricks
    st.subheader("Tips & Tricks")
    st.markdown("""
    - **Debug Panel**: Enable the debug panel to see the agents' thought processes
    - **Model Selection**: Different models have different strengths; experiment to find the best for your use case
    - **Web Search**: Enable web search to supplement information not found in your document
    - **Edit Content**: You can edit generated sections to refine or expand them
    - **Save Multiple Versions**: Generate different tutorials from the same document by changing your query
    """)

def _create_output_section(config):
    """Create output section for displaying generated tutorial.
    
    Args:
        config: Application configuration
    """
    # Check if we have content to display
    if "tutorial_content" in st.session_state and st.session_state.tutorial_content:
        tutorial_content = st.session_state.tutorial_content
        
        # Create buttons for interactive features
        col1, col2, col3 = st.columns(3)
        with col1:
            # Combine all sections for download
            full_content = "# Tutorial\n\n"
            for section_title, content in tutorial_content.items():
                full_content += f"## {section_title}\n\n{content}\n\n"
            
            # Create download button
            st.download_button(
                "Download Tutorial (Markdown)",
                full_content,
                file_name="generated_tutorial.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Copy to clipboard button (implemented with JavaScript)
            st.button("Copy to Clipboard", 
                    on_click=_copy_to_clipboard,
                    kwargs={"content": full_content},
                    use_container_width=True)
        
        with col3:
            # Button to save tutorial to outputs directory
            if st.button("Save to Outputs", use_container_width=True):
                _save_tutorial_to_outputs(config, full_content)
        
        # Divider
        st.divider()
        
        # Display tutorial sections
        for section_title, content in tutorial_content.items():
            with st.expander(section_title, expanded=True):
                st.markdown(content)
                
                # Add edit button for each section
                if st.button(f"Edit {section_title}", key=f"edit_{section_title}"):
                    st.session_state.editing_section = section_title
                    st.session_state.editing_content = content
        
        # Show edit interface if editing a section
        if "editing_section" in st.session_state and st.session_state.editing_section:
            _create_edit_interface()
    else:
        st.info("No tutorial generated yet. Upload a PDF document and specify what tutorial you want to create.")

def _create_input_section(config, model_provider, rag_pipeline, tool_registry, agent_manager):
    """Create input section for PDF upload and query.
    
    Args:
        config: Application configuration
        model_provider: Provider for accessing models
        rag_pipeline: RAG pipeline for content retrieval
        tool_registry: Registry of available tools
        agent_manager: Agent workflow manager
    """
    # Create columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload Document")
        
        # PDF upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document", 
            type=["pdf"],
            help="Upload a PDF document to generate a tutorial from"
        )
        
        # Tutorial query
        tutorial_query = st.text_area(
            "What tutorial would you like to generate?",
            height=100,
            help="Describe the tutorial you want to create from this document"
        )
        
        # Generate button
        generate_button = st.button(
            "Generate Tutorial",
            disabled=uploaded_file is None or not tutorial_query or st.session_state.processing,
            use_container_width=True
        )
        
        # Process file when button is clicked
        if generate_button and uploaded_file is not None and tutorial_query:
            # Set processing flag
            st.session_state.processing = True
            
            try:
                with st.spinner("Processing PDF..."):
                    # Save uploaded file
                    pdf_path = os.path.join(config.temp_dir, uploaded_file.name)
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process PDF
                    pdf_processor = tool_registry.get_tool("pdf_processor")
                    pdf_content = pdf_processor.process_pdf(pdf_path)
                    
                    # Initialize state for progress tracking
                    st.session_state.current_state = {
                        "query": tutorial_query,
                        "pdf_content": pdf_content,
                        "plan": [],
                        "current_section": "",
                        "tutorial_content": {},
                        "observations": [],
                        "completed": False
                    }
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Define callback for state updates
                    def state_callback(state):
                        st.session_state.current_state = state
                        
                        # Update progress
                        if "plan" in state and state["plan"]:
                            plan = state["plan"]
                            current_section = state.get("current_section", "")
                            
                            if current_section in plan:
                                progress = (plan.index(current_section) + 1) / len(plan)
                            else:
                                progress = 0
                            
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: {current_section}")
                        
                        # Force UI update
                        time.sleep(0.1)  # Small delay to ensure UI updates
                        st.experimental_rerun()
                    
                    # Run agent workflow
                    result = agent_manager.run_with_callbacks(
                        tutorial_query, 
                        pdf_content,
                        state_callback
                    )
                    
                    # Store result
                    st.session_state.tutorial_content = result.get("tutorial_content", {})
                    st.session_state.processing = False
                    
                    # Update progress
                    progress_bar.progress(1.0)
                    status_text.text("Tutorial generated successfully!")
                    
                    # Force UI update to show results tab
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.processing = False
    
    with col2:
        # Debug panel
        if st.session_state.debug_mode:
            from .debug_panel import create_debug_panel
            create_debug_panel(st.session_state.current_state)

def launch_ui(config, model_provider, rag_pipeline, tool_registry, agent_manager):
    """Launch the Streamlit UI.
    
    Args:
        config: Application configuration
        model_provider: Provider for accessing models
        rag_pipeline: RAG pipeline for content retrieval
        tool_registry: Registry of available tools
        agent_manager: Agent workflow manager
    """
    # Set page config
    st.set_page_config(
        page_title="AI Tutorial Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    _initialize_session_state()
    
    # Create UI components
    _create_sidebar(config, model_provider)
    
    # Main content
    st.title("AI Tutorial Generator")
    
    # Create tabs for input and output
    input_tab, output_tab, help_tab = st.tabs(["Create Tutorial", "Generated Output", "Help"])
    
    # Input tab
    with input_tab:
        _create_input_section(config, model_provider, rag_pipeline, tool_registry, agent_manager)
    
    # Output tab
    with output_tab:
        _create_output_section(config)
    
    # Help tab
    with help_tab:
        _create_help_section()
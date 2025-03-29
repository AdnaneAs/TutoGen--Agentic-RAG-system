"""
Model selector component for the Streamlit UI.
"""
import logging
from typing import Dict, List, Any, Optional

import streamlit as st

logger = logging.getLogger(__name__)

def create_model_selector(model_provider):
    """Create model selection component.
    
    Args:
        model_provider: Provider for accessing models
    """
    try:
        # Get available models
        available_models = model_provider.list_models()
        
        if not available_models:
            st.warning("No models found. Make sure Ollama is running and models are installed.")
            st.markdown("""
            Install models with:
            ```
            ollama pull llama3
            ollama pull nomic-embed-text
            ```
            """)
            return
        
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
            
            if "embed" in model_name or any("embed" in tag.lower() for tag in tags):
                embedding_models.append(model)
            elif "vision" in model_name or any("vision" in tag.lower() for tag in tags):
                vision_models.append(model)
            else:
                llm_models.append(model)
        
        # LLM selection
        st.subheader("LLM Model")
        llm_options = [m["name"] for m in llm_models]
        
        # Default to session state if available, otherwise use first model
        default_llm = st.session_state.get("selected_llm", llm_options[0] if llm_options else None)
        
        selected_llm = st.selectbox(
            "Select LLM Model",
            options=llm_options,
            index=llm_options.index(default_llm) if default_llm in llm_options else 0,
            disabled=not llm_options,
            help="Large Language Model for text generation"
        )
        
        if selected_llm:
            st.session_state.selected_llm = selected_llm
        
        # Embedding model selection
        st.subheader("Embedding Model")
        embedding_options = [m["name"] for m in embedding_models]
        
        # Add LLM models as fallback options for embeddings
        if not embedding_options and llm_options:
            embedding_options = llm_options
            st.info("No dedicated embedding models found. Using LLMs for embeddings.")
        
        # Default to session state if available, otherwise use first model
        default_embedding = st.session_state.get(
            "selected_embedding", 
            embedding_options[0] if embedding_options else None
        )
        
        selected_embedding = st.selectbox(
            "Select Embedding Model",
            options=embedding_options,
            index=embedding_options.index(default_embedding) if default_embedding in embedding_options else 0,
            disabled=not embedding_options,
            help="Model for creating vector embeddings"
        )
        
        if selected_embedding:
            st.session_state.selected_embedding = selected_embedding
        
        # Vision model selection (if available)
        if vision_models:
            st.subheader("Vision Model")
            vision_options = [m["name"] for m in vision_models]
            
            # Default to session state if available, otherwise use first model
            default_vision = st.session_state.get(
                "selected_vision", 
                vision_options[0] if vision_options else None
            )
            
            selected_vision = st.selectbox(
                "Select Vision Model",
                options=vision_options,
                index=vision_options.index(default_vision) if default_vision in vision_options else 0,
                help="Model for processing images"
            )
            
            if selected_vision:
                st.session_state.selected_vision = selected_vision
        
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
    
    except Exception as e:
        logger.error(f"Error creating model selector: {e}")
        st.error(f"Error loading models: {str(e)}")
        st.info("Make sure Ollama is running and accessible.")

def update_model_settings(model_provider):
    """Update model settings based on selections.
    
    Args:
        model_provider: Provider for accessing models
    """
    settings = {
        "llm_model": st.session_state.get("selected_llm"),
        "embedding_model": st.session_state.get("selected_embedding"),
        "vision_model": st.session_state.get("selected_vision"),
        "temperature": st.session_state.get("temperature", 0.7),
        "max_tokens": st.session_state.get("max_tokens", 2000)
    }
    
    return settings

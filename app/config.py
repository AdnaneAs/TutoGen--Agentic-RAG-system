"""
Configuration settings for the Agentic RAG Tutorial Generator.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class Config(BaseModel):
    """Configuration settings for the application."""
    
    # Application paths
    app_dir: str = Field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    project_root: str = Field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    data_dir: str = Field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data"))
    temp_dir: str = Field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data/temp"))
    processed_dir: str = Field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data/processed"))
    output_dir: str = Field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data/outputs"))
    
    # Vector DB settings
    vector_db_path: str = Field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data/embeddings"))
    
    # Ollama settings
    ollama_api_url: str = Field(default="http://127.0.0.1:11434")
    default_llm_model: str = Field(default="llama3")
    default_embedding_model: str = Field(default="nomic-embed-text")
    
    # Agent settings
    agent_temperature: float = Field(default=0.7)
    agent_max_tokens: int = Field(default=2000)
    
    # Web search settings
    enable_web_search: bool = Field(default=False)
    web_search_max_results: int = Field(default=5)
    
    # UI settings
    debug_mode: bool = Field(default=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.temp_dir, self.processed_dir, 
                         self.output_dir, self.vector_db_path]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def update_from_dict(self, settings_dict):
        """Update config from a dictionary."""
        for key, value in settings_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, file_path=None):
        """Save configuration to a JSON file."""
        if file_path is None:
            file_path = os.path.join(self.project_root, "config.json")
        
        with open(file_path, "w") as f:
            f.write(self.json(indent=2))
    
    @classmethod
    def load_from_file(cls, file_path):
        """Load configuration from a JSON file."""
        if not os.path.exists(file_path):
            return cls()
        
        with open(file_path, "r") as f:
            config_data = json.load(f)
        
        return cls(**config_data)

# Agentic RAG Tutorial Generator

An intelligent system that processes PDF documents to create comprehensive tutorials using agent-based architecture, RAG pipelines, and multimodal content processing.

## Features

- **PDF Processing**: Extract text, images, and tables from PDF documents
- **RAG Pipeline**: Advanced retrieval augmented generation using LlamaIndex
- **Agent-based Architecture**: Planning, research, and content generation agents with ReAct patterns
- **Web Search Integration**: Supplement content with DuckDuckGo search results
- **Local Model Support**: Use any locally installed models via Ollama
- **Interactive UI**: Streamlit interface with agent thought process visualization
- **Multimodal Support**: Handle text, images, and tables in the generated tutorials

## Architecture

This system uses a sophisticated pipeline to generate tutorials:

### 1. PDF Processing
- Extracts text, images, and tables from uploaded PDFs
- Processes structure to understand document organization
- Preserves relationships between content elements

### 2. Agents (ReAct Framework)
- **Planning Agent**: Creates the tutorial outline and structure
- **Research Agent**: Gathers relevant information for each section
- **Content Agent**: Generates the actual tutorial content
- Each agent follows a Reasoning, Action, Observation loop

### 3. RAG System
- Documents are indexed in ChromaDB vector store
- Contextual retrieval provides relevant information to agents
- Multimodal support for retrieving images and tables

### 4. Web Search (Optional)
- Supplements document content with web information
- Configurable to enable/disable as needed

### 5. Local Models via Ollama
- Uses models installed on your system through Ollama
- Configurable model selection based on task requirements

## Installation

### Prerequisites

1. Python 3.9+ with pip
2. [Ollama](https://ollama.ai/) for local model management

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-rag-tutorial.git
cd agentic-rag-tutorial
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install required models with Ollama:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### Configuration

The application uses default configurations for most settings, but you can customize:

- Model selection through the UI
- Web search enablement
- Debug mode for viewing agent processes

## Usage

1. Start the Ollama service:
```bash
ollama serve
```

2. Launch the application:
```bash
streamlit run main.py
```

3. Access the web interface at http://localhost:8501

4. Using the application:
   - Upload a PDF document
   - Enter what type of tutorial you want to generate
   - Select models and settings in the sidebar
   - Click "Generate Tutorial"
   - View the generated tutorial in the "Generated Output" tab
   - Download the markdown file for your tutorial

## Project Structure

```
agentic_rag_tutorial/
├── app/
│   ├── agents/          # ReAct agent implementations
│   │   ├── planning_agent.py
│   │   ├── research_agent.py
│   │   ├── content_agent.py
│   │   └── agent_manager.py
│   ├── models/          # Ollama model integrations
│   │   ├── ollama_client.py
│   │   └── model_registry.py
│   ├── rag/             # RAG pipeline components
│   │   ├── pipeline.py
│   │   ├── indexing.py
│   │   ├── retrieval.py
│   │   └── multimodal.py
│   ├── tools/           # Agent tools
│   │   ├── pdf_processor.py
│   │   ├── web_search.py
│   │   ├── embedding.py
│   │   ├── markdown_generator.py
│   │   └── tool_registry.py
│   ├── ui/              # Streamlit UI components
│   │   ├── main_interface.py
│   │   ├── debug_panel.py
│   │   └── model_selector.py
│   └── config.py        # Configuration settings
├── data/                # Data storage
│   ├── processed/       # Processed PDF data
│   ├── embeddings/      # Vector DB storage
│   ├── outputs/         # Generated tutorials
│   └── temp/            # Temporary files
└── main.py              # Application entry point
```

## Development

### Adding New Features

1. **New Agent**: Create a new agent in `app/agents/` that follows the ReAct pattern
2. **New Tool**: Add a tool in `app/tools/` and register it in `tool_registry.py`
3. **UI Extensions**: Enhance the Streamlit UI in `app/ui/` components

### Testing

Run basic tests with:
```bash
python -m pytest
```

## Troubleshooting

### Common Issues

1. **"No models found"**: Ensure Ollama is running and models are installed
2. **PDF processing errors**: Check PDF is not corrupted or password-protected
3. **Memory issues**: For large PDFs, try increasing your system's available memory

## License

MIT

## Credits

This project uses the following open-source libraries:
- LlamaIndex for RAG capabilities
- ChromaDB for vector storage
- LangGraph for agent workflows
- Streamlit for the user interface
- PyMuPDF and Camelot for PDF processing
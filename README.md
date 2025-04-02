# Document to Tutorial System with ReAct Agents

A comprehensive system that can convert PDF documents into well-structured educational tutorials using RAG (Retrieval-Augmented Generation) and ReAct (Reasoning and Acting) agent architecture.

## System Architecture

The system is organized into the following folders and components:

### RAG Pipeline
- **rag_pipeline.py**: Implements a RAG pipeline class using LlamaIndex and ChromaDB with functions for indexation, query, embedding, and extraction.

### Embedding
- **text_embedding.py**: Functions for text embedding using models like Nomic.
- **image_embedding.py**: Functions for image embedding using vision-language models like LLaVA.
- **tables_embedding.py**: Functions for tabular data embedding.

### Agents
- **tutorial_agent.py**: Original LangGraph-based agent implementation.
- **sequential_agent.py**: ReAct agent implementation using the Reasoning and Acting pattern (Observe, Think, Act, Repeat).

### Tools
- **pdf_tools.py**: Simplified tools for PDF extraction and summarization.
- **tutorial_tools.py**: Simplified tools for planning, writing, formatting, and improving tutorials.

### Interface
- **app.py**: Original Streamlit application.
- **app_sequential.py**: Enhanced Streamlit application with ReAct agent implementation and process logs.

## ReAct Agent Pattern

The ReAct agent follows this cyclical pattern:
1. **Observe** → Read the input & previous actions
2. **Think** → Decide the next best step and what should be done
3. **Act** → Execute a tool/action
4. **Repeat** → If needed, refine reasoning & take another action

This pattern allows for a more deliberate, step-by-step approach to tutorial generation that can be easily inspected and debugged.

## Setup and Installation

### Prerequisites
- Python 3.8+
- Ollama (for running local models)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-to-tutorial-system.git
cd document-to-tutorial-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install and set up Ollama with required models:
```bash
# Install Ollama (see https://ollama.ai/)
# Pull required models
ollama pull llama3.2:latest
ollama pull nomic-embed-text
ollama pull llava
```

## Usage

### Original App
1. Start the original Streamlit application:
```bash
streamlit run app.py
```

### ReAct Agent App (Recommended)
1. Start the enhanced ReAct application:
```bash
streamlit run app_sequential.py
```

2. Open your browser and go to http://localhost:8501

3. In the sidebar:
   - Click "Load Available Models" to fetch available models from Ollama
   - Select your preferred models for each type
   - Click "Initialize System with Selected Models"

4. On the main panel:
   - Upload a PDF document
   - Specify the tutorial goal
   - Click "Generate Tutorial"

5. The system will process the document through the following ReAct steps (observable in the "Process Logs" tab):
   - Observe the current state
   - Think about what to do next
   - Execute the chosen action
   - Repeat until the tutorial is complete

6. You can download the final tutorial in Markdown format or explore the PDF content and tutorial plan in the other tabs.

## System Components in Detail

### RAG Pipeline (rag_pipeline.py)
- Indexes and retrieves documents using vector similarity
- Supports different embedding models for text, images, and tables
- Integrates with LlamaIndex and ChromaDB

### Embedding Modules
- Specialized embeddings for different content types
- Handles model selection dynamically from available Ollama models

### ReAct Agent (sequential_agent.py)
Implements the ReAct pattern:
- **Observe**: Takes in the current state and forms an observation
- **Think**: Reasons about what to do next
- **Act**: Executes the chosen tool/action
- **Repeat**: Continues the cycle until the task is complete

### Tutorial Tools (tutorial_tools.py)
- **SimpleTutorialPlanner**: Plans tutorial structure based on PDF content
- **SimpleTutorialWriter**: Writes tutorial sections using RAG to find relevant content
- **SimpleTutorialFormatter**: Formats tutorial with proper markdown and visual elements
- **SimpleTutorialImprover**: Identifies and implements improvements to the tutorial

### PDF Tools (pdf_tools.py)
- Functions for extracting text, images, tables from PDFs
- Summarizes PDF content for use in tutorial planning
- Extracts structural elements for better organization

## Advanced Configuration

### Custom Models
You can use custom models by adding them to Ollama. For example:
```bash
ollama pull mistral:latest
ollama pull phi:latest
```

### ChromaDB Configuration
By default, ChromaDB stores data in the chroma_db folder. You can modify this in the RAG pipeline:
```python
from rag.rag_pipeline import RAGPipeline
rag = RAGPipeline(persist_dir="path/to/custom/chroma_dir")
```

## Troubleshooting

### Common Issues

1. **Model loading errors**:
   - Check if Ollama service is running
   - Verify that the required models are pulled into Ollama
   - Try restarting the Ollama service

2. **PDF extraction issues**:
   - Complex PDFs with heavy formatting might not extract properly
   - Try with a simpler PDF to verify functionality
   - Check permissions on the uploaded_pdfs directory

3. **Memory issues**:
   - Large PDFs might cause memory issues, especially with image extraction
   - Consider increasing RAM allocation for Python or reducing PDF size

### Logs

Check the application logs for more detailed error information. Logs are displayed in the terminal and in the "Process Logs" tab.

## Contributing

Contributions are welcome! Here's how you can contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

Please make sure to update tests as appropriate and follow the coding style.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- LlamaIndex for the RAG components
- LangGraph for agent framework inspiration
- Streamlit for the web interface
- Ollama for local model hosting

## Requirements

The system requires the following Python packages:
- streamlit
- llama-index
- chromadb
- langchain
- langgraph
- pymupdf (for PDF processing)
- requests
- pillow
- pandas
- numpy

## Future Improvements

Some potential enhancements for the future:
- Adding more specialized extractors for different document types
- Implementing benchmarking for different model combinations
- Adding support for multi-document tutorial generation
- Creating a visualization module for the ReAct decision process
- Supporting more output formats (HTML, PDF, slides)
- Implementing user feedback mechanisms to improve tutorial quality
- Adding collaborative editing features
- Supporting multilingual tutorial generation
# RAG MCP Application

This project combines a Retrieval-Augmented Generation (RAG) system with the Model Context Protocol (MCP) to create a powerful, modular AI application. It features a dedicated MCP server (`rag_server.py`) that exposes RAG capabilities and a weather tool, and a client UI (`client_ui.py`) that uses an orchestrator LLM to interact with these tools.

## Project Structure

- `rag-mcp-app/`
    - `data/`: Directory for your PDF documents to be indexed by the RAG system.
    - `chroma_db/`: Directory where the ChromaDB vector store will be persisted.
    - `rag_server.py`: The MCP server that hosts the RAG and weather tools.
    - `client_ui.py`: The client application with a Gradio UI that orchestrates LLM calls and tool usage.
    - `ingest.py`: A script to load and index your PDF documents into the vector database.
    - `.env.example`: An example configuration file. Copy this to `.env` to configure the application.
    - `requirements.txt`: Lists all project dependencies.
    - `README.md`: This file.

## Getting Started

### Prerequisites

*   **Python 3.11+**: Ensure you have Python installed.
*   **Ollama**: Install Ollama from [ollama.ai](https://ollama.ai/) and ensure it's running.
*   **Ollama Models**: Pull the necessary models for the client's orchestrator LLM and the RAG LLM (if using Ollama). The defaults are `qwen3:1.7b`.
    ```bash
    ollama pull qwen3:1.7b
    ```
*   **Google API Key**: If using Gemini, set your `GOOGLE_API_KEY` in the `.env` file.

### Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd rag-mcp-app
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Configure the Application:**
    - Copy the example environment file: `cp .env.example .env`
    - Edit the `.env` file to set your desired configuration (e.g., models, ports, API keys).

### Data Preparation

1.  **Populate the `data/` directory**: Place your PDF documents into the `rag-mcp-app/data/` directory.

2.  **Run the Ingestion Script**: This needs to be run *before* you start the RAG server for the first time, or whenever you add new documents to the `data/` directory.
    ```bash
    python ingest.py
    ```

### Running the Application

Activate your virtual environment in your terminal. The client application is designed to start the MCP server as a background process, so you only need to run one command:

```bash
uv run python client_ui.py --mcp-server rag_server.py
```

The client will:
1.  Start the `rag_server.py` script.
2.  Connect to the server.
3.  Launch the Gradio UI, which will be accessible at the port specified in your `.env` file (e.g., `http://127.0.0.1:3000`).

### Example Usage

*   **Ask a question about your documents:** "What is the main topic of the documents?"
*   **Ask about the weather:** "What's the weather like in London?"

## To Do
- remove RAG LLM. Redundant and accidental.
- Make Chroma Vector DB a resource
- Make access to embedding model and Vector DB a tool, when lacks current knowledge
- Review Architecture, Code
- Consider adding web search after that
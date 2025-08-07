# RAG MCP Application

This project demonstrates a powerful, modular AI application using the Model Context Protocol (MCP). The architecture follows a clean agent-tool-resource model: a central orchestrator LLM acts as the agent, consuming tools provided by a lean MCP server to access various resources.

The core components are:
-   **`client_ui.py`**: A Gradio-based client that houses the single orchestrator LLM. This agent is responsible for all reasoning, including deciding when to use tools and generating final responses based on tool outputs.
-   **`rag_server.py`**: A lightweight MCP server that provides tools to access resources. It does not contain any LLM. The available tools are:
    -   `search_knowledge_base`: Accesses a ChromaDB vector database (the resource) to retrieve relevant information.
    -   `get_weather`: Accesses an external weather API (the resource).

This separation of concerns makes the system highly modular and easy to extend.

## Project Structure

-   `rag-mcp-app/`
    -   `data/`: Directory for your PDF documents to be indexed.
    -   `chroma_db/`: Directory where the ChromaDB vector store is persisted.
    -   `rag_server.py`: The MCP server that provides tools.
    -   `client_ui.py`: The client application with the orchestrator LLM and Gradio UI.
    -   `ingest.py`: Script to index PDF documents into the vector database.
    -   `.env`: Your local configuration file.
    -   `requirements.txt`: Project dependencies.

## Getting Started

### Prerequisites

*   **Python 3.13+**: Ensure you have a compatible Python version installed.
*   **Ollama**: Install Ollama from [ollama.ai](https://ollama.ai/) and ensure it's running.
*   **Ollama Model**: Pull the model for the orchestrator LLM. The default is `qwen3:1.7b`.
    ```bash
    ollama pull qwen3:1.7b
    ```
*   **Google API Key**: For document embeddings, set your `GOOGLE_API_KEY` in the `.env` file.

### Installation

1.  **Clone the repository and navigate into it.**

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv .venv
    # On Windows: .\.venv\Scripts\activate
    # On macOS/Linux: source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Configure the Application:**
    -   Copy the example environment file: `cp .env.example .env`
    -   Edit the `.env` file to set your `GOOGLE_API_KEY` and any other desired configurations (e.g., model, port).

### Data Preparation

1.  **Populate the `data/` directory**: Place your PDF documents into the `rag-mcp-app/data/` directory.

2.  **Run the Ingestion Script**: This must be run before starting the application for the first time, or whenever you update the documents.
    ```bash
    uv run python ingest.py
    ```

### Running the Application

Activate your virtual environment. The client application starts the MCP server as a background process, so you only need to run one command:

```bash
cd rag-mcp-app; .\.venv\Scripts\activate; uv run python client_ui.py --mcp-server rag_server.py
  
uv run python client_ui.py --mcp-server rag_server.py
```

The client will start the server, connect to it, and launch the Gradio UI. Access it in your browser at the configured port (e.g., `http://127.0.0.1:3000`).

### Example Usage

*   **Ask a question about your documents:** "What is the main topic of the documents?"
*   **Ask about the weather:** "What's the weather like in London?"

## Project Status

The initial refactoring is complete. The architecture now correctly implements the agent-tool-resource model.

-   [x] **Review Architecture, Code**: The architecture has been reviewed and refactored for clarity and modularity.
-   [x] **Remove RAG LLM**: The redundant LLM has been removed from the server.
-   [x] **Make Chroma Vector DB a resource**: The vector DB is now treated as a resource, accessed via a dedicated tool.
-   [x] **Make access to Vector DB a tool**: The `search_knowledge_base` tool provides this functionality.
-   [ ] **Consider adding web search**: The new architecture makes this easy. A new tool can be added to `rag_server.py` to enable web search capabilities.

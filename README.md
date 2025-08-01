# RAG MCP Application

This project combines a Retrieval-Augmented Generation (RAG) system with the Model Context Protocol (MCP) to create a powerful, modular AI application. It features a dedicated MCP server (`rag_server.py`) that exposes RAG capabilities and a weather tool, and a client UI (`client_ui.py`) that uses an orchestrator LLM to interact with these tools.

## Project Structure

- `rag-mcp-app/`
    - `data/`: Directory for your PDF documents to be indexed by the RAG system.
    - `chroma_db/`: Directory where the ChromaDB vector store will be persisted.
    - `rag_server.py`: The MCP server that hosts the RAG and weather tools.
    - `client_ui.py`: The client application with a Gradio UI that orchestrates LLM calls and tool usage.
    - `ingest.py`: A script to load and index your PDF documents into the vector database.
    - `requirements.txt`: Lists all project dependencies.
    - `README.md`: This file.

## Getting Started

### Prerequisites

*   **Python 3.11+**: Ensure you have Python installed.
*   **Ollama**: Install Ollama from [ollama.ai](https://ollama.ai/) and ensure it's running.
*   **Ollama Model**: Pull the `qwen3:1.7b` model:
    ```bash
    ollama pull qwen3:1.7b
    ```
*   **OpenAI API Key**: Set your OpenAI API key as an environment variable (e.g., in a `.env` file). The RAG system uses OpenAI for embeddings and potentially for the LLM if not using Ollama for RAG.

### Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd rag-mcp-app
    ```
    *(Note: If you are following along with the development process, you would have already created this directory and copied files into it.)*

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

1.  **Populate the `data/` directory**: Place your PDF documents into the `rag-mcp-app/data/` directory.

### Running the Application

You will need to run two processes: the MCP server and the client UI.

**1. Start the MCP Server:**

Open a new terminal, activate your virtual environment, and run:
```bash
python rag_server.py
```
This will start the MCP server, making the `get_weather` and `get_rag_response` tools available.

**2. Start the Client UI:**

Open another terminal, activate your virtual environment, and run:
```bash
python client_ui.py --mcp-server rag_server.py --model qwen3:1.7b
```
This command connects the client UI to the MCP server and specifies the orchestrator LLM.

The client UI will launch in your browser. You can then interact with the chatbot, asking questions that might trigger the RAG system or the weather tool.

### Example Usage

*   **Ask a question about your documents:** "What is the main topic of the documents?"
*   **Ask about the weather:** "What's the weather like in London?"

---
**Note:** The `ingest.py` script needs to be run *before* you start the RAG server for the first time, or whenever you add new documents to the `data/` directory.
```bash
# Run this in a separate terminal before starting the server/client if you have new data
python ingest.py

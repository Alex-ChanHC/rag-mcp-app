# rag_server.py
import argparse
import asyncio
import urllib
from contextlib import AsyncExitStack
from mcp.server.fastmcp import FastMCP
from typing import List, Dict

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variables (e.g., for GOOGLE_API_KEY)
load_dotenv()

# Configuration for RAG
DATA_PATH = os.environ.get("DATA_PATH", "data")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "models/embedding-001")
RETRIEVER_NUM_RESULTS = int(os.environ.get("RETRIEVER_NUM_RESULTS", 5))

# Initialize embeddings model using Google Generative AI
# Ensure your GOOGLE_API_KEY is set in your .env file
embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDINGS_MODEL)

# Initialize the vector store
# Ensure the chroma_db directory is created and populated by ingest.py before running the server.
# We create the directory here to ensure it exists if ingest.py hasn't run yet.
os.makedirs(CHROMA_PATH, exist_ok=True)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Set up the retriever
retriever = vector_store.as_retriever(search_kwargs={'k': RETRIEVER_NUM_RESULTS})

# Initialize FastMCP server
# The name "rag-weather-server" is used for identifying the server.
mcp_server = FastMCP("rag-weather-server")

# Register the 'get_weather' tool
@mcp_server.tool()
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.

    This tool fetches weather information from an external API (wttr.in).

    Args:
      city (str): The name of the city for which to retrieve the weather.

    Returns:
      str: A string describing the current weather in the specified city,
           e.g., "Sunny +20°C".
    """
    # Construct the URL for the weather API, encoding the city name properly.
    # The format parameter requests only the current condition and temperature.
    contents = urllib.request.urlopen(f'https://wttr.in/{urllib.parse.quote_plus(city)}?format=%C+%t').read()
    # Decode the response from bytes to a UTF-8 string and return it.
    return contents.decode('utf-8')

# Register the 'search_knowledge_base' tool
@mcp_server.tool()
async def search_knowledge_base(query: str) -> str:
    """
    Searches the knowledge base for information relevant to the query.

    This tool retrieves relevant text chunks from the Chroma vector database
    based on the user's query.

    Args:
        query (str): The user's question or search term.

    Returns:
        str: A string containing the concatenated content of the retrieved documents,
             to be used as context for the orchestrator LLM.
    """
    print(f"Searching knowledge base for: '{query}'")
    # Retrieve relevant chunks
    docs = await retriever.ainvoke(query)

    # Concatenate the content of the retrieved documents
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"
    
    print(f"Found knowledge: {knowledge[:200]}...") # Log snippet of the knowledge
    return knowledge

# Main function to run the server
if __name__ == "__main__":
    print("Starting RAG MCP server...")
    # 'transport='stdio'' means the server will communicate via standard input/output,
    # which is how the MCP client connects to it.
    mcp_server.run(transport='stdio')

# rag_server.py
import argparse
import asyncio
import urllib
from contextlib import AsyncExitStack
import ollama
from ollama import ChatResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from rich.console import Console
from rich.markdown import Markdown
from typing import Optional, List, Dict, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.llms import Ollama # Import Ollama for local models
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables (e.g., for GOOGLE_API_KEY)
load_dotenv()

# Configuration for RAG
DATA_PATH = os.environ.get("DATA_PATH", "data")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "models/embedding-001")
RAG_LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen3:1.7b")
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

# Register the 'get_rag_response' tool
@mcp_server.tool()
async def get_rag_response(query: str, history: List[Dict[str, str]]) -> str:
    """
    Answers questions based on provided knowledge using RAG.

    This tool retrieves relevant information from a vector database and uses
    an LLM (Gemini or Ollama) to generate an answer based solely on that information.

    Args:
        query (str): The user's question.
        history (List[Dict[str, str]]): The conversation history.

    Returns:
        str: The LLM's answer.
    """
    # Retrieve relevant chunks based on the question asked
    # Use async invoke if available and the tool is async
    docs = await retriever.ainvoke(query)

    # Add all the chunks to 'knowledge'
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    # Construct the RAG prompt
    rag_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge,
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the provided knowledge.
        """),
        ("user", """The question: {question}

        Conversation history: {history}

        The knowledge: {knowledge}
        """)
    ])

    # Create the RAG chain
    rag_chain = (
        {"question": RunnablePassthrough(), "history": lambda _: history, "knowledge": lambda _: knowledge}
        | rag_prompt_template
        | rag_llm
        | StrOutputParser()
    )

    # Execute the chain and return the answer
    answer = await rag_chain.ainvoke(query)
    return answer

# Main function to parse arguments and set up LLM
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--llm-provider", default=os.environ.get("RAG_LLM_PROVIDER", "ollama"), choices=["ollama", "gemini"],
                   help="Choose the LLM provider for RAG: 'ollama' (default) or 'gemini'.")
    args = p.parse_args()

    # Initialize rag_llm based on the chosen provider
    if args.llm_provider == "gemini":
        rag_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
        print("Using Google Gemini for RAG LLM.")
    else: # Default to ollama
        rag_llm = Ollama(model=RAG_LLM_MODEL, temperature=0.5)
        print(f"Using Ollama ({RAG_LLM_MODEL}) for RAG LLM.")

    # Run the MCP server
    # 'transport='stdio'' means the server will communicate via standard input/output,
    # which is how the MCP client connects to it.
    # FastMCP handles running async tools when the tool function is defined as async.
    mcp_server.run(transport='stdio')

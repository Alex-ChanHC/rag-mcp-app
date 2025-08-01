# ingest.py
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import os

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
# DATA_PATH is relative to the script's location, which will be inside rag-mcp-app
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Ensure the data directory exists and is populated
if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
    print(f"Error: Data directory '{DATA_PATH}' not found or is empty.")
    print("Please ensure you have copied your PDF files into the 'data' directory within 'rag-mcp-app'.")
    exit(1)

# Initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Ensure the chroma_db directory exists
os.makedirs(CHROMA_PATH, exist_ok=True)

# Initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Loading the PDF document
print(f"Loading documents from: {DATA_PATH}")
loader = PyPDFDirectoryLoader(DATA_PATH)

try:
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents.")
except Exception as e:
    print(f"Error loading documents: {e}")
    exit(1)

# Splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# Creating the chunks
print("Splitting documents into chunks...")
chunks = text_splitter.split_documents(raw_documents)
print(f"Created {len(chunks)} chunks.")

# Creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Adding chunks to vector store
print(f"Adding {len(chunks)} chunks to ChromaDB at '{CHROMA_PATH}'...")
try:
    vector_store.add_documents(documents=chunks, ids=uuids)
    print("Successfully added documents to ChromaDB.")
except Exception as e:
    print(f"Error adding documents to ChromaDB: {e}")
    exit(1)

print("Ingestion complete.")

import os

# Google Gemini (GenAI) Configuration
GENAI_API_KEY = os.environ["GOOGLE_API_KEY"]

# Pinecone Configuration
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
PINECONE_DIMENSION = int(os.environ["PINECONE_DIMENSION"])
PINECONE_METRIC = os.environ["PINECONE_METRIC"]
PINECONE_CLOUD = os.environ["PINECONE_CLOUD"]
PINECONE_REGION = os.environ["PINECONE_REGION"]

# Default LLM and embedding model names
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"

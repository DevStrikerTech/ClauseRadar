# ClauseRadar

ClauseRadar is an AI-powered Streamlit application for searching contract clauses in PDF documents. It leverages Google
Gemini for text embeddings and Pinecone for efficient vector search. Users can upload one or more PDF contracts, enter
any keyword or phrase, and retrieve the most relevant clauses across all indexed documents.

## Features

- **PDF Ingestion**: Upload multiple PDF contracts and automatically extract text.
- **Dynamic Query**: Enter any keyword or free-text phrase to search across all contracts.
- **Semantic Search**: Uses Google Gemini embeddings to understand query intent and Pinecone for fast nearest-neighbor
  lookup.
- **Interactive UI**: Displays results in a responsive table and bar chart (compatible with both light and dark modes).
- **Full Snippet Preview**: Hover over truncated snippets to see the entire clause and download the original PDF.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/clause-radar.git
   cd clause-radar

2. **install dependencies using Poetry (Python 3.11+)**:
   ```bash
   poetry install
   ```
3. **Set API keys in ~/.streamlit/secrets.toml**:
   ```toml
   [google]
   GENAI_API_KEY = "your_google_gemini_api_key"
   
   [pinecone]
   PINECONE_API_KEY = "your_pinecone_api_key"
   ```

## Usage

**Run the Streamlit app**:

   ```bash
   poetry run streamlit run app.py
   ```
1. **Index Contracts**
   - In the sidebar, upload one or more PDF contracts.
   - Enter comma-separated keywords or phrases (e.g., Effective Date, Payment Terms, Termination).
   - Click Index Contracts to extract, embed, and store clause snippets in Pinecone.


2. **Search**
   - In the sidebar, type any keyword or free-text query (e.g., confidential information).
   - Select the number of results to return.
   - Click Search to retrieve and display top matches in the main panel.
   
## Project Structure
```plaintaxt
ClauseRadar/
├── pyproject.toml
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── app.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── embedding_service.py
    ├── pinecone_client.py
    ├── contract_recommender.py
    ├── ui.py
    └── utils.py
```
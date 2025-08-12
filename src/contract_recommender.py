import io
import re
import hashlib
from typing import Any, Dict, List

import pdfplumber

from src.embedding_service import EmbeddingService
from src.pinecone_client import PineconeClient


class ContractRecommender:
    """
    Orchestrates:
      1) PDF ingestion & text extraction
      2) Dynamic snippet mining based on user-provided keywords
      3) Embedding via Google Gemini
      4) Upserting/querying Pinecone
    """

    def __init__(
            self,
            embedding_service: EmbeddingService,
            pinecone_client: PineconeClient,
    ) -> None:
        """
        Initialize the ContractRecommender.

        Args:
            embedding_service: Instance of EmbeddingService for embeddings.
            pinecone_client: Instance of PineconeClient for vector operations.
        """
        self.embedder = embedding_service
        self.pinecone = pinecone_client
        # In‐memory cache: snippet_hash → embedding vector
        self._cache: Dict[str, List[float]] = {}

    @staticmethod
    def _hash_text(text: str) -> str:
        """
        Compute a SHA-256 hash of the given text for deduplication.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_all_text(pdf_file: io.BytesIO) -> str:
        """
        Given a file-like object for a PDF, extract all text sequentially.
        """
        full_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += "\n" + text
        return full_text

    @staticmethod
    def _find_snippet(full_text: str, keyword: str) -> str:
        """
        Search the full_text for the first occurrence of the keyword (case-insensitive).
        If found, return ~300 characters around that match for context.
        If not found exactly, return an empty string.
        """
        pattern = rf"(?i){re.escape(keyword)}"
        match = re.search(pattern, full_text)
        if not match:
            return ""
        start = max(0, match.start() - 50)
        end = min(len(full_text), match.end() + 250)
        return full_text[start:end].strip()

    def index_contracts(
            self,
            contract_files: Dict[str, io.BytesIO],
            keywords: List[str],
    ) -> None:
        """
        For each (contract_id, file_bytes):
          1. Extract text.
          2. For each keyword, find snippet.
          3. Embed snippet (with caching).
          4. Upsert to Pinecone under ID "contract_id::keyword".
        """
        vectors_to_upsert: List[tuple[str, List[float], Dict[str, Any]]] = []

        for contract_id, file_bytes in contract_files.items():
            full_text = self._extract_all_text(file_bytes)
            for keyword in keywords:
                snippet = self._find_snippet(full_text, keyword)
                if not snippet:
                    continue

                snippet_hash = self._hash_text(snippet)
                if snippet_hash in self._cache:
                    emb = self._cache[snippet_hash]
                else:
                    emb = self.embedder.embed_text(snippet)
                    self._cache[snippet_hash] = emb

                vector_id = f"{contract_id}::{keyword}"
                metadata = {
                    "contract_id": contract_id,
                    "keyword": keyword,
                    "snippet": snippet,
                }
                vectors_to_upsert.append((vector_id, emb, metadata))

        if vectors_to_upsert:
            self.pinecone.upsert_batch(vectors_to_upsert)

    def recommend(
            self,
            user_query: str,
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Embed the user_query, query Pinecone, and return a list of matches:
          [
            {
              "contract_id": ...,
              "keyword": ...,
              "score": ...,
              "snippet": ...,
            },
            ...
          ]

        Results are sorted descending by score.
        """
        query_emb = self.embedder.embed_text(user_query)
        raw_matches = self.pinecone.query(query_emb, top_k=top_k)

        results: List[Dict[str, Any]] = []
        for match in raw_matches:
            meta = match["metadata"]
            results.append(
                {
                    "contract_id": meta["contract_id"],
                    "keyword": meta["keyword"],
                    "score": match["score"],
                    "snippet": meta["snippet"],
                }
            )
        return results

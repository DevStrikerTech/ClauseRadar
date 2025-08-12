from typing import List, Optional

import google.generativeai as genai

from src.config import GENAI_API_KEY, EMBEDDING_MODEL


class EmbeddingService:
    """
    Service to generate text embeddings using Gemini (GenAI).
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: Optional[str] = None,
    ) -> None:
        """
        Initialize the EmbeddingService.

        Args:
            api_key: API key for Google Gemini. If None, uses GENAI_API_KEY from config.
            model: Embedding model name. If None, uses EMBEDDING_MODEL from config.
        """
        self.api_key = api_key or GENAI_API_KEY
        genai.configure(api_key=self.api_key)
        self.model = model or EMBEDDING_MODEL

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: The input string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document",
        )
        return response.get("embedding", [])

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Batch‐embed a list of texts by calling embed_text on each.

        Args:
            texts: List of input strings.

        Returns:
            A list of embedding vectors (one per input string).
        """
        return [self.embed_text(t) for t in texts]

from typing import Any, Dict, List

from pinecone import Pinecone, ServerlessSpec


class PineconeClient:
    """
    Client for interacting with a Pinecone index using the new Pinecone SDK.

    Attributes:
        client: The Pinecone client instance.
        index: The Pinecone Index object.
    """

    def __init__(
            self,
            api_key: str,
            cloud: str,
            region: str,
            index_name: str,
            dimension: int,
            metric: str,
    ) -> None:
        """
        Initialize Pinecone and create the index if it does not exist.

        Args:
            api_key: Pinecone API key.
            cloud: Pinecone cloud provider (e.g., "gcp", "aws").
            region: Pinecone region (e.g., "us-east1-gcp").
            index_name: Name of the index to use or create.
            dimension: Dimensionality of the embedding vectors.
            metric: Distance metric for similarity search ("cosine", "euclidean", etc.).
        """
        # Create a Pinecone client instance
        self.client = Pinecone(api_key=api_key)

        # Check existing indexes
        existing_indexes = self.client.list_indexes().names()
        if index_name not in existing_indexes:
            # Create a new index with serverless spec
            spec = ServerlessSpec(cloud=cloud, region=region)
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=spec,
            )

        # Get a handle to the index
        self.index = self.client.Index(index_name)

    def upsert_batch(
            self,
            vectors: List[tuple[str, List[float], Dict[str, Any]]],
            batch_size: int = 100,
    ) -> None:
        """
        Upsert a batch of vectors into Pinecone.

        Args:
            vectors: List of tuples (vector_id, vector_values, metadata_dict).
            batch_size: Number of vectors to upsert in each batch.
        """
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i: i + batch_size]
            self.index.upsert(vectors=batch)

    def query(
            self,
            query_vector: List[float],
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query the index for the nearest neighbors to the given vector.

        Args:
            query_vector: Embedding vector to query.
            top_k: Number of top results to retrieve.

        Returns:
            A list of matches, each containing 'id', 'score', and 'metadata'.
        """
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = []
        for match in response["matches"]:
            matches.append(
                {
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match["metadata"],
                }
            )
        return matches

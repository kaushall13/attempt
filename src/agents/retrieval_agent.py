from typing import List, Dict, Any
from src.services.embedding_service import EmbeddingService
from src.vector_stores.dummy_vector_store import DummyVectorStore

class RetrievalAgent:
    def __init__(self, embedding_service: EmbeddingService, vector_store: DummyVectorStore):
        """
        Initializes the RetrievalAgent.

        Args:
            embedding_service: An instance of EmbeddingService to generate query vectors.
            vector_store: An instance of a vector store (e.g., DummyVectorStore) to search for documents.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves documents relevant to a given query.

        Args:
            query: The input query string.
            top_k: The number of top results to return.

        Returns:
            A list of dictionaries, where each dictionary represents a retrieved document
            and contains its text, metadata, and similarity score.
        """
        # 1. Encode the query to get a vector
        query_vector = self.embedding_service.encode(query)

        # 2. Search the vector store for similar documents
        search_results = self.vector_store.search(query_vector, top_k=top_k)

        # 3. Format and return the results
        return self._format_results(search_results)

    def _format_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formats the search results into the desired output structure.
        """
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "score": result["score"],
                }
            )
        return formatted_results

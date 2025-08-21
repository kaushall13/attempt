import numpy as np
from typing import List, Dict, Any
from src.services.embedding_service import EmbeddingService

class DummyVectorStore:
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initializes the DummyVectorStore.

        Args:
            embedding_service: An instance of EmbeddingService to generate embeddings.
        """
        self.embedding_service = embedding_service
        self.documents = self._create_dummy_documents()

    def _create_dummy_documents(self) -> List[Dict[str, Any]]:
        """
        Creates a few dummy documents and generates their embeddings.
        """
        texts = [
            "The sky is blue.",
            "The sun is bright.",
            "The cat sat on the mat.",
        ]

        docs = []
        for text in texts:
            embedding = self.embedding_service.encode(text)
            docs.append(
                {
                    "text": text,
                    "metadata": {"source": "dummy.txt"},
                    "embedding": embedding,
                }
            )
        return docs

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a similarity search using cosine similarity.
        """
        query_np = np.array(query_vector)

        results = []
        for doc in self.documents:
            doc_np = np.array(doc["embedding"])

            # Since vectors are L2-normalized, cosine similarity is the dot product
            sim = np.dot(query_np, doc_np)

            results.append(
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": sim,
                }
            )

        # Sort by score descending and return top_k
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

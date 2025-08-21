import numpy as np
from typing import List, Dict, Any

class DummyVectorStore:
    def __init__(self):
        # In a real vector store, you would initialize a connection to a database
        # or load an index from a file.
        self.documents = self._create_dummy_documents()

    def _create_dummy_documents(self) -> List[Dict[str, Any]]:
        """Creates a few dummy documents for demonstration."""
        return [
            {
                "text": "The sky is blue.",
                "metadata": {"source": "nature.txt"},
                "embedding": np.random.rand(1536).tolist(),
            },
            {
                "text": "The sun is bright.",
                "metadata": {"source": "nature.txt"},
                "embedding": np.random.rand(1536).tolist(),
            },
            {
                "text": "The cat sat on the mat.",
                "metadata": {"source": "animals.txt"},
                "embedding": np.random.rand(1536).tolist(),
            },
        ]

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Simulates a similarity search in the vector store.
        In a real implementation, this would use an efficient search algorithm
        like HNSW or IVF. Here, we just return dummy results.
        """
        results = []
        for doc in self.documents:
            # Simulate a similarity score (e.g., cosine similarity)
            # For this dummy implementation, we'll just use a random score.
            score = np.random.uniform(0.7, 1.0)
            results.append(
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score,
                }
            )

        # Sort by score descending and return top_k
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

from qdrant_client import QdrantClient, models

class VectorStore:
    def __init__(self, collection_name="my_collection"):
        self.client = QdrantClient(":memory:")  # Use in-memory storage for simplicity
        self.collection_name = collection_name
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )

    def upsert(self, vectors, payloads):
        """
        Upserts vectors and their payloads into the collection.
        """
        points = [
            models.PointStruct(id=i, vector=vector, payload=payload)
            for i, (vector, payload) in enumerate(zip(vectors, payloads))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query_vector, limit=5):
        """
        Searches for similar vectors in the collection.
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
        )
        return search_result

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient as QC
from qdrant_client.http import models

load_dotenv()

class QdrantClient:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the environment variables.")

        try:
            self.client = QC(url=qdrant_url, api_key=qdrant_api_key)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

    def create_collection(self, name: str, vector_size: int):
        """
        Creates a new collection in Qdrant if it does not exist.
        """
        try:
            if not self.client.collection_exists(collection_name=name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
                print(f"Collection '{name}' created successfully.")
            else:
                print(f"Collection '{name}' already exists.")
        except Exception as e:
            print(f"Error creating collection '{name}': {e}")

    def add_documents(self, collection: str, docs: list[str], embeddings: list[list[float]], metadata: list[dict]):
        """
        Upserts documents with their embeddings and metadata into a collection.
        """
        if not all([docs, embeddings, metadata]):
            raise ValueError("docs, embeddings, and metadata must be provided.")

        if len(docs) != len(embeddings) or len(docs) != len(metadata):
            raise ValueError("The lengths of docs, embeddings, and metadata must be the same.")

        points = []
        for doc, embedding, meta in zip(docs, embeddings, metadata):
            point_id = str(uuid.uuid4())
            payload = {"document": doc, **meta}
            points.append(models.PointStruct(id=point_id, vector=embedding, payload=payload))

        try:
            self.client.upsert(collection_name=collection, points=points, wait=True)
            print(f"Upserted {len(docs)} documents into collection '{collection}'.")
        except Exception as e:
            print(f"Error upserting documents into collection '{collection}': {e}")

    def search(self, collection: str, query_vector: list[float], limit: int = 10, score_threshold: float = 0.7):
        """
        Searches for similar vectors in a collection.
        """
        try:
            hits = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            if not hits:
                print("No results found.")
            return hits
        except Exception as e:
            print(f"Error searching in collection '{collection}': {e}")
            return []

    def delete_collection(self, name: str):
        """
        Deletes a collection from Qdrant.
        """
        try:
            self.client.delete_collection(collection_name=name)
            print(f"Collection '{name}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting collection '{name}': {e}")

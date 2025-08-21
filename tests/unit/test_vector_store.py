import pytest
from app.vector_store import VectorStore
import numpy as np

def test_vector_store_upsert_and_search():
    """
    Tests that the VectorStore can upsert and search for vectors.
    """
    vector_store = VectorStore(collection_name="test_collection")

    # Upsert a vector
    vector = np.random.rand(384).tolist()
    payload = {"text": "test document"}
    vector_store.upsert([vector], [payload])

    # Search for the vector
    search_result = vector_store.search(vector, limit=1)

    assert len(search_result) == 1
    assert search_result[0].payload["text"] == "test document"

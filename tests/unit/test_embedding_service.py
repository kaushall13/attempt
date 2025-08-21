import pytest
from app.embedding_service import EmbeddingService

def test_create_embedding():
    """
    Tests that the create_embedding method returns a vector of the correct size.
    """
    embedding_service = EmbeddingService()
    embedding = embedding_service.create_embedding("test text")
    assert isinstance(embedding, list)
    assert len(embedding) == 384

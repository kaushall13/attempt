import pytest
from unittest.mock import MagicMock
import numpy as np

@pytest.fixture
def mock_embedding_service(mocker):
    """Mocks the EmbeddingService."""
    mock = MagicMock()
    mock.create_embedding.return_value = np.random.rand(384).tolist()
    return mock

@pytest.fixture
def mock_vector_store(mocker):
    """Mocks the VectorStore."""
    mock = MagicMock()
    mock.search.return_value = [
        MagicMock(payload={"text": "The sky is blue."})
    ]
    return mock

@pytest.fixture
def mock_groq(mocker):
    """Mocks the Groq client."""
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "The sky is indeed blue."

    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.return_value = mock_completion

    mocker.patch('app.agents.Groq', return_value=mock_groq_client)
    return mock_groq_client

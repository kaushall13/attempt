import pytest
from app.agents import RAGAgent

def test_rag_agent_answer(mock_embedding_service, mock_vector_store, mock_groq):
    """
    Tests the RAGAgent's answer method using mocked services.
    """
    agent = RAGAgent(
        groq_api_key="fake-api-key",
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )

    question = "What color is the sky?"
    answer = agent.answer(question)

    # Verify that the mock services were called
    mock_embedding_service.create_embedding.assert_called_once_with(question)
    mock_vector_store.search.assert_called_once()

    # Verify that the Groq client was called to generate a completion
    mock_groq.chat.completions.create.assert_called_once()

    # Verify that the answer is what we expect from the mock
    assert answer == "The sky is indeed blue."

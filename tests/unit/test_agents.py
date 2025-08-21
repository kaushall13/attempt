import pytest
from app.agents import RAGAgent
from unittest.mock import MagicMock

def test_rag_agent_answer_with_context(mock_embedding_service, mock_vector_store, mock_groq):
    """
    Tests the RAGAgent's answer method with context.
    """
    agent = RAGAgent(
        groq_api_key="fake-api-key",
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )

    question = "What color is the sky?"
    answer = agent.answer(question)

    mock_embedding_service.create_embedding.assert_called_once_with(question)
    mock_vector_store.search.assert_called_once()
    mock_groq.chat.completions.create.assert_called_once()

    args, kwargs = mock_groq.chat.completions.create.call_args
    system_message = kwargs['messages'][0]['content']
    assert "The sky is blue." in system_message
    assert answer == "The sky is indeed blue."

def test_rag_agent_answer_no_context(mock_embedding_service, mock_vector_store, mock_groq):
    """
    Tests the RAGAgent's answer method when no context is found.
    """
    mock_vector_store.search.return_value = []
    agent = RAGAgent(
        groq_api_key="fake-api-key",
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )

    question = "What is the meaning of life?"
    agent.answer(question)

    mock_groq.chat.completions.create.assert_called_once()
    args, kwargs = mock_groq.chat.completions.create.call_args
    system_message = kwargs['messages'][0]['content']
    assert "could not find any relevant context" in system_message

def test_rag_agent_context_deduplication(mock_embedding_service, mock_vector_store, mock_groq):
    """
    Tests that the context passed to the LLM is deduplicated.
    """
    mock_vector_store.search.return_value = [
        MagicMock(payload={"text": "The sky is blue."}),
        MagicMock(payload={"text": "The sky is blue."}),
        MagicMock(payload={"text": "The grass is green."}),
    ]
    agent = RAGAgent(
        groq_api_key="fake-api-key",
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )

    agent.answer("What color is the sky and grass?")

    mock_groq.chat.completions.create.assert_called_once()
    args, kwargs = mock_groq.chat.completions.create.call_args
    system_message = kwargs['messages'][0]['content']
    assert system_message.count("The sky is blue.") == 1
    assert "The grass is green." in system_message

def test_rag_agent_token_budgeting(mock_embedding_service, mock_vector_store, mock_groq):
    """
    Tests that the context is truncated based on the token limit.
    """
    long_text = "This is a very long text. " * 500
    mock_vector_store.search.return_value = [
        MagicMock(payload={"text": "Short text."}),
        MagicMock(payload={"text": long_text}),
    ]
    # Set a small token limit to easily test truncation
    agent = RAGAgent(
        groq_api_key="fake-api-key",
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        max_context_tokens=10
    )

    agent.answer("A question")

    mock_groq.chat.completions.create.assert_called_once()
    args, kwargs = mock_groq.chat.completions.create.call_args
    system_message = kwargs['messages'][0]['content']

    # The context should only contain "Short text." because the long_text exceeds the token limit.
    assert "Short text." in system_message
    assert long_text not in system_message

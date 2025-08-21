import pytest
from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore
from app.agents import RAGAgent

def test_end_to_end_rag_pipeline(mock_groq):
    """
    Tests the RAG pipeline end-to-end, from embedding to answer generation.
    Mocks the external Groq API.
    """
    # 1. Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore(collection_name="end_to_end_test_1")

    # 2. Add data to the vector store
    documents = [
        "The sky is blue.",
        "The grass is green.",
        "The sun is bright.",
    ]
    embeddings = [embedding_service.create_embedding(doc) for doc in documents]
    payloads = [{"text": doc} for doc in documents]
    vector_store.upsert(embeddings, payloads)

    # 3. Initialize the RAG agent
    rag_agent = RAGAgent(
        groq_api_key="fake-api-key",
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # 4. Ask a question
    question = "What color is the sky?"
    answer = rag_agent.answer(question)

    # 5. Verify the answer
    assert answer == "The sky is indeed blue."

    # Verify that the Groq client was called
    mock_groq.chat.completions.create.assert_called_once()
    args, kwargs = mock_groq.chat.completions.create.call_args
    system_message = kwargs['messages'][0]['content']
    assert "The sky is blue" in system_message

def test_end_to_end_no_context(mock_groq):
    """
    Tests the RAG pipeline end-to-end when no context is found.
    """
    # 1. Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore(collection_name="end_to_end_test_2")

    # 2. RAG agent
    rag_agent = RAGAgent(
        groq_api_key="fake-api-key",
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # 3. Ask a question
    rag_agent.answer("A question with no context")

    # 4. Verify that the correct system message is used
    mock_groq.chat.completions.create.assert_called_once()
    args, kwargs = mock_groq.chat.completions.create.call_args
    system_message = kwargs['messages'][0]['content']
    assert "could not find any relevant context" in system_message

def test_end_to_end_deduplication(mock_groq):
    """
    Tests that the RAG pipeline deduplicates context.
    """
    # 1. Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore(collection_name="end_to_end_test_3")

    # 2. Add data to the vector store
    documents = [
        "The sky is blue.",
        "The sky is blue.",
        "The grass is green.",
    ]
    embeddings = [embedding_service.create_embedding(doc) for doc in documents]
    payloads = [{"text": doc} for doc in documents]
    vector_store.upsert(embeddings, payloads)

    # 3. RAG agent
    rag_agent = RAGAgent(
        groq_api_key="fake-api-key",
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # 4. Ask a question
    rag_agent.answer("A question")

    # 5. Verify that the context is deduplicated
    mock_groq.chat.completions.create.assert_called_once()
    args, kwargs = mock_groq.chat.completions.create.call_args
    system_message = kwargs['messages'][0]['content']
    assert system_message.count("The sky is blue.") == 1
    assert "The grass is green." in system_message

def test_end_to_end_token_budgeting(mock_groq):
    """
    Tests that the RAG pipeline respects the token budget.
    """
    # 1. Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore(collection_name="end_to_end_test_4")

    # 2. Add data to the vector store
    long_text = "This is a very long text. " * 100
    documents = [
        "This is a short text.",
        long_text,
    ]
    embeddings = [embedding_service.create_embedding(doc) for doc in documents]
    payloads = [{"text": doc} for doc in documents]
    vector_store.upsert(embeddings, payloads)

    # 3. RAG agent with a small token limit
    rag_agent = RAGAgent(
        groq_api_key="fake-api-key",
        embedding_service=embedding_service,
        vector_store=vector_store,
        max_context_tokens=10
    )

    # 4. Ask a question
    rag_agent.answer("A question")

    # 5. Verify that the context is truncated
    mock_groq.chat.completions.create.assert_called_once()
    args, kwargs = mock_groq.chat.completions.create.call_args
    system_message = kwargs['messages'][0]['content']
    assert "This is a short text." in system_message
    assert long_text not in system_message

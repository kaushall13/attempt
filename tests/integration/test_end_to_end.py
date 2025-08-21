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
    vector_store = VectorStore(collection_name="end_to_end_test")

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
    # You could add more specific assertions here about the context passed to Groq
    # For example, check that the context contained "The sky is blue."
    args, kwargs = mock_groq.chat.completions.create.call_args
    system_message = kwargs['messages'][0]['content']
    assert "The sky is blue" in system_message

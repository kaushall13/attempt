import os
from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore
from app.agents import RAGAgent

def main():
    # 1. Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    # 2. Add some data to the vector store
    documents = [
        "The sky is blue.",
        "The grass is green.",
        "The sun is bright.",
    ]
    embeddings = [embedding_service.create_embedding(doc) for doc in documents]
    payloads = [{"text": doc} for doc in documents]
    vector_store.upsert(embeddings, payloads)

    # 3. Initialize the RAG agent
    # Make sure to set the GROQ_API_KEY environment variable
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    rag_agent = RAGAgent(
        groq_api_key=groq_api_key,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # 4. Ask a question
    question = "What color is the sky?"
    answer = rag_agent.answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

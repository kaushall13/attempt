from src.agents.retrieval_agent import RetrievalAgent
from src.services.embedding_service import EmbeddingService
from src.vector_stores.dummy_vector_store import DummyVectorStore

def main():
    """
    An example script to demonstrate the usage of the RetrievalAgent.
    """
    # 1. Initialize the components
    embedding_service = EmbeddingService()
    vector_store = DummyVectorStore()
    retrieval_agent = RetrievalAgent(
        embedding_service=embedding_service,
        vector_store=vector_store
    )

    # 2. Define a query and retrieve documents
    query = "What is nature?"
    retrieved_docs = retrieval_agent.retrieve(query, top_k=2)

    # 3. Print the results
    print(f"Query: '{query}'\n")
    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. Text: '{doc['text']}'")
        print(f"     Metadata: {doc['metadata']}")
        print(f"     Score: {doc['score']:.4f}\n")

if __name__ == "__main__":
    main()

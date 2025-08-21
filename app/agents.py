from groq import Groq
from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore

class RAGAgent:
    def __init__(self, groq_api_key, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def answer(self, question):
        """
        Answers a question using a RAG (Retrieval-Augmented Generation) approach.
        """
        # 1. Create an embedding for the question
        question_embedding = self.embedding_service.create_embedding(question)

        # 2. Search for relevant context in the vector store
        search_results = self.vector_store.search(question_embedding)
        context = " ".join([result.payload["text"] for result in search_results])

        # 3. Generate an answer using Groq API
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Use the following context to answer the question:\n\n{context}",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content

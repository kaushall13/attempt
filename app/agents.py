import tiktoken
from groq import Groq
from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore

class RAGAgent:
    def __init__(self, groq_api_key, embedding_service: EmbeddingService, vector_store: VectorStore, max_context_tokens=4096):
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.max_context_tokens = max_context_tokens
        # Using cl100k_base as a common tokenizer. For Llama3, a specific tokenizer would be better.
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def answer(self, question):
        """
        Answers a question using a RAG (Retrieval-Augmented Generation) approach.
        """
        # 1. Create an embedding for the question
        question_embedding = self.embedding_service.create_embedding(question)

        # 2. Search for relevant context in the vector store
        search_results = self.vector_store.search(question_embedding)

        # 3. Process and budget the context
        if not search_results:
            context = ""
            system_message = "You are a helpful assistant. Please answer the question to the best of your ability as I could not find any relevant context."
        else:
            # Deduplicate documents
            unique_documents = list(dict.fromkeys([result.payload["text"] for result in search_results]))

            # Token budgeting
            context_str = ""
            current_token_count = 0
            for doc in unique_documents:
                doc_tokens = self.tokenizer.encode(doc)
                if current_token_count + len(doc_tokens) <= self.max_context_tokens:
                    context_str += doc + " "
                    current_token_count += len(doc_tokens)
                else:
                    break
            context = context_str.strip()
            system_message = f"You are a helpful assistant. Use the following context to answer the question:\\n\\n{context}"


        # 4. Generate an answer using Groq API
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content

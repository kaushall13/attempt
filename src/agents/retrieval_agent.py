class RetrievalAgent:
    def retrieve_documents(self, plan: str) -> list[str]:
        print(f"Retrieval agent: Retrieving documents for plan '{plan}'")
        return ["doc1.txt", "doc2.txt"]

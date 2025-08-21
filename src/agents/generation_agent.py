class GenerationAgent:
    def generate_response(self, documents: list[str], query: str) -> str:
        print(f"Generation agent: Generating response for query '{query}' using documents: {documents}")
        return f"Response based on {documents}"

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GenerationAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def _create_prompt(self, query, context_docs):
        context = "\n".join(context_docs)
        prompt = f"""
        You are a helpful assistant. Based on the following context, please answer the user's query.
        Provide citations from the context in the format [citation: index].

        Context:
        {context}

        Query: {query}
        """
        return prompt.strip()

    def generate_response(self, retrieved_docs, original_query):
        prompt = self._create_prompt(original_query, retrieved_docs)

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-8b-8192",
            )

            response_text = chat_completion.choices[0].message.content
            return self._post_process(response_text)
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Error generating response."

    def _post_process(self, response_text):
        # Simple validation and citation formatting
        if not response_text:
            return "No response generated."

        # This is a very basic citation formatting example.
        # In a real-world scenario, this would be more robust.
        import re
        citations = re.findall(r'\[citation: (\d+)\]', response_text)

        formatted_response = response_text
        if citations:
            formatted_response += "\n\nCitations:"
            for i, c in enumerate(set(citations)):
                formatted_response += f"\n[{i+1}] Source {c}"

        return formatted_response

if __name__ == '__main__':
    # Example Usage
    retrieved_docs = [
        "Doc 1: The sky is blue.",
        "Doc 2: The grass is green.",
    ]
    original_query = "What color is the sky and the grass?"

    agent = GenerationAgent()
    response = agent.generate_response(retrieved_docs, original_query)
    print(response)

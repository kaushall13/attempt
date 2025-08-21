import logging
from agents.planning_agent import PlanningAgent
from agents.retrieval_agent import RetrievalAgent
from agents.generation_agent import GenerationAgent
from agents.memory_manager import MemoryManager

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RAGOrchestrator:
    def __init__(self):
        """Initializes all the agents."""
        self.planning_agent = PlanningAgent()
        self.retrieval_agent = RetrievalAgent()
        self.generation_agent = GenerationAgent()
        self.memory_manager = MemoryManager()
        logging.info("RAGOrchestrator initialized with all agents.")

    def process_query(self, query: str) -> str:
        """
        Processes a user query by coordinating the agents.
        1. Planning agent analyzes query.
        2. Retrieval agent gets relevant docs.
        3. Generation agent creates response.
        4. Memory manager stores interaction.
        """
        logging.info(f"Processing query: '{query}'")
        try:
            # 1. Planning agent analyzes query
            plan = self.planning_agent.analyze_query(query)
            logging.info(f"Plan created: {plan}")

            # 2. Retrieval agent gets relevant docs
            documents = self.retrieval_agent.retrieve_documents(plan)
            logging.info(f"Documents retrieved: {documents}")

            # 3. Generation agent creates response
            response = self.generation_agent.generate_response(documents, query)
            logging.info(f"Response generated: {response}")

            # 4. Memory manager stores interaction
            self.memory_manager.store_interaction(query, response)
            logging.info("Interaction stored in memory.")

            return response
        except Exception as e:
            logging.error(f"An error occurred during query processing: {e}", exc_info=True)
            return "I'm sorry, but I encountered an error while processing your request."

if __name__ == "__main__":
    """A simple command-line interface for testing the RAG orchestrator."""
    orchestrator = RAGOrchestrator()
    print("RAG Orchestrator CLI is running. Type 'exit' to quit.")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Exiting CLI.")
            break

        response = orchestrator.process_query(user_query)
        print(f"Bot: {response}")

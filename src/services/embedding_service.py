import numpy as np

class EmbeddingService:
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        self.dimension = 1536  # Example dimension for text-embedding-ada-002

    def encode(self, text: str) -> list[float]:
        """
        Simulates encoding a text string into a vector embedding.
        In a real implementation, this would call an actual embedding model.
        """
        # Simulate a random embedding for demonstration purposes
        return np.random.rand(self.dimension).tolist()

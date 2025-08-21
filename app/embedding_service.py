import numpy as np

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name

    def create_embedding(self, text):
        """
        Creates an embedding for the given text.
        In a real application, this would use a sentence-transformer model.
        For this example, we'll return a fixed-size random vector.
        """
        return np.random.rand(384).tolist()

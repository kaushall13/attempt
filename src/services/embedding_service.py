import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the EmbeddingService using a SentenceTransformer model.

        Args:
            model_name: The name of the sentence-transformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> list[float]:
        """
        Encodes a text string into a vector embedding using the loaded model
        and applies L2 normalization.

        Args:
            text: The text to encode.

        Returns:
            A L2-normalized vector embedding as a list of floats.
        """
        # Encode the text
        embedding = self.model.encode(text, convert_to_tensor=True)

        # L2 normalization
        normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

        return normalized_embedding.tolist()

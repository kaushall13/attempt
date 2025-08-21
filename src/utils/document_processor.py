import os
from typing import List, Dict, Any

class DocumentProcessor:
    """
    A class to process text documents, including loading, chunking, and metadata extraction.
    """

    def load_text_file(self, filepath: str) -> str:
        """
        Loads text content from a file.

        Args:
            filepath: The path to the text file.

        Returns:
            The content of the file as a string.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        allowed_extensions = ['.txt', '.md']
        if not any(filepath.endswith(ext) for ext in allowed_extensions):
            raise ValueError(f"Unsupported file type. Supported types are: {', '.join(allowed_extensions)}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200, strategy: str = 'fixed') -> List[str]:
        """
        Splits text into chunks based on a specified strategy.

        Args:
            text: The input text to be chunked.
            chunk_size: The desired size of each chunk.
            overlap: The number of characters or sentences to overlap.
            strategy: The chunking strategy ('fixed' or 'sentence').

        Returns:
            A list of text chunks.
        """
        if not isinstance(text, str):
            raise TypeError("Input 'text' must be a string.")
        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")
        if overlap < 0:
            raise ValueError("Overlap must be a non-negative integer.")

        if strategy == 'fixed':
            return self._fixed_size_chunking(text, chunk_size, overlap)
        elif strategy == 'sentence':
            return self._sentence_aware_chunking(text, chunk_size, overlap)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Supported strategies are 'fixed' and 'sentence'.")

    def _fixed_size_chunking(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size for 'fixed' strategy.")

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def _sentence_aware_chunking(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        # NOTE: For this strategy, 'overlap' is interpreted as the number of overlapping sentences.
        import re
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]
        if not sentences:
            return []

        chunks = []
        i = 0
        while i < len(sentences):
            chunk_sentences = []
            current_len = 0

            # Start building a chunk from index i
            j = i
            while j < len(sentences):
                sentence = sentences[j]
                # Add 1 for the space between sentences
                if current_len + len(sentence) + (1 if chunk_sentences else 0) > chunk_size and chunk_sentences:
                    break
                chunk_sentences.append(sentence)
                current_len += len(sentence) + (1 if len(chunk_sentences) > 1 else 0)
                j += 1

            chunks.append(" ".join(chunk_sentences))

            # If the last chunk was formed, break
            if j == len(sentences):
                break

            # Move the main index 'i' forward to create the next chunk with overlap
            i = max(i + 1, j - overlap)

        return chunks

    def extract_metadata(self, source: str) -> Dict[str, Any]:
        """
        Extracts basic metadata from a file path or a string.

        Args:
            source: The file path or the string content.

        Returns:
            A dictionary containing metadata.
        """
        if os.path.isfile(source):
            return {
                'source': source,
                'size': os.path.getsize(source),
                'type': 'file'
            }
        else:
            return {
                'source': 'string',
                'size': len(source),
                'type': 'string'
            }

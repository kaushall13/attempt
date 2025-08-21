import unittest
from unittest.mock import patch, MagicMock
import os

# Set dummy environment variables for testing
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "test_key"

from src.retrieval.vector_store import QdrantClient
from qdrant_client.http import models

class TestQdrantClient(unittest.TestCase):

    @patch('src.retrieval.vector_store.QC')
    def test_init_success(self, mock_qdrant_client):
        """Test successful initialization of QdrantClient."""
        client = QdrantClient()
        self.assertIsNotNone(client)
        mock_qdrant_client.assert_called_with(url="http://localhost:6333", api_key="test_key")

    @patch.dict(os.environ, {"QDRANT_URL": "", "QDRANT_API_KEY": ""})
    def test_init_missing_env_vars(self):
        """Test initialization with missing environment variables."""
        with self.assertRaises(ValueError):
            QdrantClient()

    @patch('src.retrieval.vector_store.QC')
    def test_create_collection_new(self, mock_qdrant_client):
        """Test the create_collection method for a new collection."""
        mock_client_instance = MagicMock()
        mock_client_instance.collection_exists.return_value = False
        mock_qdrant_client.return_value = mock_client_instance

        client = QdrantClient()
        client.create_collection("test_collection", 128)

        mock_client_instance.create_collection.assert_called_with(
            collection_name="test_collection",
            vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE)
        )

    @patch('src.retrieval.vector_store.QC')
    def test_create_collection_existing(self, mock_qdrant_client):
        """Test the create_collection method for an existing collection."""
        mock_client_instance = MagicMock()
        mock_client_instance.collection_exists.return_value = True
        mock_qdrant_client.return_value = mock_client_instance

        client = QdrantClient()
        client.create_collection("test_collection", 128)

        mock_client_instance.create_collection.assert_not_called()

    @patch('src.retrieval.vector_store.QC')
    def test_add_documents(self, mock_qdrant_client):
        """Test the add_documents method."""
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance

        client = QdrantClient()
        docs = ["doc1", "doc2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadata = [{"meta1": "val1"}, {"meta2": "val2"}]

        client.add_documents("test_collection", docs, embeddings, metadata)

        self.assertEqual(mock_client_instance.upsert.call_count, 1)
        args, kwargs = mock_client_instance.upsert.call_args
        self.assertEqual(kwargs['collection_name'], "test_collection")
        self.assertEqual(len(kwargs['points']), 2)

    @patch('src.retrieval.vector_store.QC')
    def test_search_with_threshold(self, mock_qdrant_client):
        """Test the search method with a score_threshold."""
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.search.return_value = "search_results"

        client = QdrantClient()
        results = client.search("test_collection", [0.1, 0.2], score_threshold=0.8)

        mock_client_instance.search.assert_called_with(
            collection_name="test_collection",
            query_vector=[0.1, 0.2],
            limit=10,
            score_threshold=0.8
        )
        self.assertEqual(results, "search_results")

    @patch('src.retrieval.vector_store.QC')
    def test_search_no_results(self, mock_qdrant_client):
        """Test the search method when no results are found."""
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.search.return_value = []

        client = QdrantClient()
        results = client.search("test_collection", [0.1, 0.2])

        self.assertEqual(results, [])

    @patch('src.retrieval.vector_store.QC')
    def test_delete_collection(self, mock_qdrant_client):
        """Test the delete_collection method."""
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance

        client = QdrantClient()
        client.delete_collection("test_collection")

        mock_client_instance.delete_collection.assert_called_with(collection_name="test_collection")

if __name__ == '__main__':
    unittest.main()

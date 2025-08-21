import unittest
import os
from src.utils.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DocumentProcessor()
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)

        self.txt_file = os.path.join(self.test_dir, "test.txt")
        with open(self.txt_file, "w") as f:
            f.write("This is a test file. It has multiple sentences. This is the third sentence.")

        self.md_file = os.path.join(self.test_dir, "test.md")
        with open(self.md_file, "w") as f:
            f.write("# Markdown File\n\nThis is a test markdown file.")

    def tearDown(self):
        os.remove(self.txt_file)
        os.remove(self.md_file)
        os.rmdir(self.test_dir)

    def test_load_text_file_txt(self):
        content = self.processor.load_text_file(self.txt_file)
        self.assertEqual(content, "This is a test file. It has multiple sentences. This is the third sentence.")

    def test_load_text_file_md(self):
        content = self.processor.load_text_file(self.md_file)
        self.assertEqual(content, "# Markdown File\n\nThis is a test markdown file.")

    def test_load_text_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.load_text_file("non_existent_file.txt")

    def test_load_text_file_unsupported_type(self):
        unsupported_file = os.path.join(self.test_dir, "test.unsupported")
        with open(unsupported_file, "w") as f:
            f.write("unsupported")
        with self.assertRaises(ValueError):
            self.processor.load_text_file(unsupported_file)
        os.remove(unsupported_file)

    def test_fixed_size_chunking(self):
        text = "This is a long string that needs to be chunked into smaller pieces."
        chunks = self.processor.chunk_text(text, chunk_size=20, overlap=5, strategy='fixed')
        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[0], "This is a long strin")
        self.assertEqual(chunks[1], "string that needs to")

    def test_sentence_aware_chunking(self):
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        # With chunk_size=70, s1 and s2 should be in the first chunk.
        # With overlap=1, the next chunk should start with s2.
        chunks = self.processor.chunk_text(text, chunk_size=60, overlap=1, strategy='sentence')
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "This is the first sentence. This is the second sentence.")
        self.assertEqual(chunks[1], "This is the second sentence. This is the third sentence.")

    def test_extract_metadata_file(self):
        metadata = self.processor.extract_metadata(self.txt_file)
        self.assertEqual(metadata['source'], self.txt_file)
        self.assertEqual(metadata['type'], 'file')
        self.assertGreater(metadata['size'], 0)

    def test_extract_metadata_string(self):
        text = "This is a string."
        metadata = self.processor.extract_metadata(text)
        self.assertEqual(metadata['source'], 'string')
        self.assertEqual(metadata['type'], 'string')
        self.assertEqual(metadata['size'], len(text))

if __name__ == '__main__':
    unittest.main()

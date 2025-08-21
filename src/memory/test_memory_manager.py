import unittest
from .memory_manager import MemoryManager

class TestMemoryManager(unittest.TestCase):

    def setUp(self):
        self.memory = MemoryManager()
        self.session_id = self.memory.start_session()

    def test_start_session(self):
        self.assertIsNotNone(self.session_id)
        self.assertIn(self.session_id, self.memory.sessions)

    def test_add_interaction(self):
        self.memory.add_interaction(self.session_id, "Hello", "Hi there!")
        self.assertEqual(len(self.memory.sessions[self.session_id]), 1)
        self.assertEqual(self.memory.sessions[self.session_id][0]['query'], "Hello")

    def test_get_recent_context(self):
        self.memory.add_interaction(self.session_id, "What is AI?", "AI is...")
        self.memory.add_interaction(self.session_id, "Tell me a joke", "Why did the chicken cross the road?")

        context = self.memory.get_recent_context(self.session_id, n=1)
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]['query'], "Tell me a joke")

    def test_get_recent_context_relevance(self):
        self.memory.add_interaction(self.session_id, "What is your name?", "I am a bot.")
        self.memory.add_interaction(self.session_id, "What can you do?", "I can answer questions.")
        self.memory.add_interaction(self.session_id, "Do you know about python?", "Yes, Python is a programming language.")

        context = self.memory.get_recent_context(self.session_id, n=5, relevance_keywords=["python"])
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]['query'], "Do you know about python?")

    def test_clear_memory(self):
        self.memory.add_interaction(self.session_id, "Hi", "Hello")
        self.assertTrue(self.memory.clear_memory(self.session_id))
        self.assertNotIn(self.session_id, self.memory.sessions)
        self.assertFalse(self.memory.clear_memory("non_existent_session"))

if __name__ == '__main__':
    unittest.main()

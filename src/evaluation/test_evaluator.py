import unittest
import time
from src.evaluation.evaluator import SimpleEvaluator

class TestSimpleEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = SimpleEvaluator()

    def test_evaluate_retrieval(self):
        retrieved_docs = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant_docs = ['doc1', 'doc3', 'doc6']
        metrics = self.evaluator.evaluate_retrieval(retrieved_docs, relevant_docs, k=5)
        self.assertAlmostEqual(metrics['top_k_accuracy'], 2/5)
        self.assertAlmostEqual(metrics['coverage'], 2/3)

    def test_evaluate_generation(self):
        response = "This is a test response."
        citations = ['[1]', '[2]']
        metrics = self.evaluator.evaluate_generation(response, citations)
        self.assertEqual(metrics['response_length'], 5)
        self.assertEqual(metrics['citation_count'], 2)

    def test_measure_system(self):
        start_time = time.time()
        time.sleep(0.1) # Simulate work
        end_time = time.time()
        metrics = self.evaluator.measure_system(start_time, end_time, error=False)
        self.assertGreater(metrics['response_time'], 0.1)
        self.assertEqual(metrics['error_rate'], 0)

        metrics_error = self.evaluator.measure_system(start_time, end_time, error=True)
        self.assertEqual(metrics_error['error_rate'], 1)

    def test_generate_report(self):
        # Retrieval data
        self.evaluator.evaluate_retrieval(['doc1', 'doc2'], ['doc1'], k=2)
        self.evaluator.evaluate_retrieval(['doc3', 'doc4'], ['doc5'], k=2)

        # Generation data
        self.evaluator.evaluate_generation("Response one.", ['[1]'])
        self.evaluator.evaluate_generation("Response two has two citations.", ['[2]', '[3]'])

        # System data
        self.evaluator.measure_system(time.time(), time.time() + 0.2, error=False)
        self.evaluator.measure_system(time.time(), time.time() + 0.3, error=True)

        report = self.evaluator.generate_report()

        self.assertIn('retrieval', report)
        self.assertIn('generation', report)
        self.assertIn('system', report)

        self.assertAlmostEqual(report['retrieval']['top_k_accuracy'], (0.5 + 0.0) / 2)
        self.assertAlmostEqual(report['retrieval']['coverage'], (1.0 + 0.0) / 2)

        self.assertAlmostEqual(report['generation']['response_length'], (2 + 5) / 2)
        self.assertAlmostEqual(report['generation']['citation_count'], (1 + 2) / 2)

        self.assertGreater(report['system']['response_time'], 0)
        self.assertAlmostEqual(report['system']['error_rate'], 0.5)

if __name__ == '__main__':
    unittest.main()

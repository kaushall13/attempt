import time
from collections import defaultdict

class SimpleEvaluator:
    """
    A simple evaluator for RAG systems.
    """
    def __init__(self):
        self.results = {
            'retrieval': [],
            'generation': [],
            'system': []
        }

    def evaluate_retrieval(self, retrieved_docs, relevant_docs, k=5):
        """
        Evaluates the retrieval component.

        Args:
            retrieved_docs (list): A list of retrieved document IDs.
            relevant_docs (list): A list of ground truth relevant document IDs.
            k (int): The number of top documents to consider for accuracy.
        """
        # Top-k accuracy
        top_k_retrieved = retrieved_docs[:k]
        hits = len(set(top_k_retrieved) & set(relevant_docs))
        top_k_accuracy = hits / k if k > 0 else 0

        # Coverage
        coverage = len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs) if relevant_docs else 0

        retrieval_metrics = {
            'top_k_accuracy': top_k_accuracy,
            'coverage': coverage
        }
        self.results['retrieval'].append(retrieval_metrics)
        return retrieval_metrics

    def evaluate_generation(self, response, citations):
        """
        Evaluates the generation component.

        Args:
            response (str): The generated response.
            citations (list): A list of citation markers in the response.
        """
        # Response length
        response_length = len(response.split())

        # Citation count
        citation_count = len(citations)

        generation_metrics = {
            'response_length': response_length,
            'citation_count': citation_count
        }
        self.results['generation'].append(generation_metrics)
        return generation_metrics

    def measure_system(self, start_time, end_time, error=False):
        """
        Measures system performance.

        Args:
            start_time (float): The start time of the operation.
            end_time (float): The end time of the operation.
            error (bool): Whether an error occurred.
        """
        response_time = end_time - start_time
        error_rate = 1 if error else 0

        system_metrics = {
            'response_time': response_time,
            'error_rate': error_rate
        }
        self.results['system'].append(system_metrics)
        return system_metrics

    def generate_report(self):
        """
        Generates a summary report of all evaluations.
        """
        report = defaultdict(lambda: defaultdict(float))

        for category, metrics_list in self.results.items():
            if not metrics_list:
                continue

            num_samples = len(metrics_list)
            for metrics in metrics_list:
                for key, value in metrics.items():
                    report[category][key] += value

        for category, metrics in report.items():
            num_samples = len(self.results[category])
            for key in metrics:
                report[category][key] /= num_samples

        return dict(report)

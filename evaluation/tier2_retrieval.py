"""
Tier 2: Retrieval Precision Evaluators.

Validates RAG/vector search quality and appropriateness.
"""

from typing import Tuple, Dict, List, Any


class RetrievalEvaluator:
    """Evaluates retrieval quality for RAG systems."""

    @staticmethod
    def evaluate_vector_search_usage(
        vector_calls: List[Dict[str, Any]], test_case: Dict
    ) -> Tuple[float, str]:
        """
        Evaluate if vector search was used appropriately.

        Args:
            vector_calls: List of vector_search calls extracted from agent output
            test_case: Test case dict with 'need_vector' flag

        Returns:
            (score, detail_message)
        """
        needs_vector = test_case.get("need_vector", False)
        vector_called = len(vector_calls) > 0

        if needs_vector and vector_called:
            return 1.0, "Vector search called appropriately"
        elif not needs_vector and not vector_called:
            return 1.0, "Vector search not needed"
        elif needs_vector and not vector_called:
            return 0.0, "Vector search should have been called"
        else:
            return 0.5, "Vector search called but may not be necessary"

    @staticmethod
    def evaluate_retrieval_relevance(
        vector_calls: List[Dict[str, Any]], test_case: Dict
    ) -> Tuple[float, str]:
        """
        Evaluate relevance of vector search queries by checking if they contain
        expected search terms defined in the test case.

        Args:
            vector_calls: List of vector_search calls extracted from agent output
            test_case: Test case dict with 'expected_search_terms' field

        Returns:
            (score, detail_message)
        """
        expected_terms = test_case.get("expected_search_terms", [])
        if not expected_terms:
            return 1.0, "No expected search terms defined"

        if not vector_calls:
            return 0.0, "No vector search performed but expected terms required"

        # Check if any vector search query contains expected terms
        for call in vector_calls:
            query = call.get("query", "").lower()
            matched_terms = [term for term in expected_terms if term.lower() in query]
            if matched_terms:
                return (
                    1.0,
                    f"Found expected terms in query '{call['query']}': {matched_terms}",
                )

        return 0.0, f"Missing expected terms in all queries: {expected_terms}"

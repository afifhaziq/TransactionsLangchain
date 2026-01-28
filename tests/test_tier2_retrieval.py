"""
Tests for the Tier 2 Retrieval Evaluator.

These tests validate vector search usage evaluation and
retrieval relevance scoring.
"""


class TestVectorSearchUsage:
    """Test vector search usage evaluation logic."""

    def test_passes_when_vector_needed_and_called(self, retrieval_evaluator):
        """Should pass when vector search is needed and called."""
        vector_calls = [{"query": "restaurants", "n_results": 15}]
        test_case = {"need_vector": True}

        score, detail = retrieval_evaluator.evaluate_vector_search_usage(
            vector_calls, test_case
        )

        assert score == 1.0
        assert "appropriately" in detail.lower()

    def test_passes_when_vector_not_needed_and_not_called(self, retrieval_evaluator):
        """Should pass when vector search not needed and not called."""
        vector_calls = []
        test_case = {"need_vector": False}

        score, detail = retrieval_evaluator.evaluate_vector_search_usage(
            vector_calls, test_case
        )

        assert score == 1.0
        assert "not needed" in detail.lower()

    def test_fails_when_vector_needed_but_not_called(self, retrieval_evaluator):
        """Should fail when vector search is needed but not called."""
        vector_calls = []
        test_case = {"need_vector": True}

        score, detail = retrieval_evaluator.evaluate_vector_search_usage(
            vector_calls, test_case
        )

        assert score == 0.0
        assert "should have been called" in detail.lower()

    def test_partial_score_when_unnecessary_call(self, retrieval_evaluator):
        """Should give partial score when vector called but not needed."""
        vector_calls = [{"query": "something", "n_results": 5}]
        test_case = {"need_vector": False}

        score, detail = retrieval_evaluator.evaluate_vector_search_usage(
            vector_calls, test_case
        )

        assert score == 0.5
        assert "not be necessary" in detail.lower()

    def test_handles_missing_need_vector_flag(self, retrieval_evaluator):
        """Should handle test case without need_vector flag (defaults to False)."""
        vector_calls = []
        test_case = {}  # No need_vector flag

        score, detail = retrieval_evaluator.evaluate_vector_search_usage(
            vector_calls, test_case
        )

        assert score == 1.0  # Defaults to not needed, and not called

    def test_multiple_vector_calls_counted(self, retrieval_evaluator):
        """Multiple vector calls should still count as 'called'."""
        vector_calls = [
            {"query": "restaurants", "n_results": 15},
            {"query": "groceries", "n_results": 10},
        ]
        test_case = {"need_vector": True}

        score, detail = retrieval_evaluator.evaluate_vector_search_usage(
            vector_calls, test_case
        )

        assert score == 1.0


class TestRetrievalRelevance:
    """Test retrieval relevance evaluation logic."""

    def test_passes_with_matching_term(self, retrieval_evaluator):
        """Should pass when query contains expected search term."""
        vector_calls = [{"query": "restaurants", "n_results": 15}]
        test_case = {"expected_search_terms": ["restaurants"]}

        score, detail = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        assert score == 1.0
        assert "found expected terms" in detail.lower()

    def test_passes_with_partial_match(self, retrieval_evaluator):
        """Should pass when query contains part of expected term."""
        vector_calls = [{"query": "grocery stores", "n_results": 15}]
        test_case = {"expected_search_terms": ["grocery"]}

        score, detail = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        assert score == 1.0

    def test_fails_with_wrong_term(self, retrieval_evaluator):
        """Should fail when query doesn't contain expected terms."""
        vector_calls = [{"query": "electronics", "n_results": 15}]
        test_case = {"expected_search_terms": ["restaurants"]}

        score, detail = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        assert score == 0.0
        assert "missing expected terms" in detail.lower()

    def test_passes_when_no_terms_expected(self, retrieval_evaluator):
        """Should pass when no search terms are expected."""
        vector_calls = []
        test_case = {"expected_search_terms": []}

        score, detail = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        assert score == 1.0

    def test_handles_missing_expected_terms(self, retrieval_evaluator):
        """Should handle test case without expected_search_terms."""
        vector_calls = [{"query": "anything", "n_results": 5}]
        test_case = {}  # No expected_search_terms

        score, detail = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        assert score == 1.0

    def test_case_insensitive_matching(self, retrieval_evaluator):
        """Should match terms case-insensitively."""
        vector_calls = [{"query": "RESTAURANTS", "n_results": 15}]
        test_case = {"expected_search_terms": ["restaurants"]}

        score, detail = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        assert score == 1.0

    def test_fails_when_no_vector_calls_but_terms_expected(self, retrieval_evaluator):
        """Should fail when terms expected but no vector search performed."""
        vector_calls = []
        test_case = {"expected_search_terms": ["restaurants"]}

        score, detail = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        assert score == 0.0

    def test_any_matching_call_sufficient(self, retrieval_evaluator):
        """Should pass if any vector call contains expected term."""
        vector_calls = [
            {"query": "electronics", "n_results": 15},
            {"query": "restaurants", "n_results": 10},
        ]
        test_case = {"expected_search_terms": ["restaurants"]}

        score, detail = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        assert score == 1.0

    def test_multiple_expected_terms_any_match(self, retrieval_evaluator):
        """Should pass if any of multiple expected terms is found."""
        vector_calls = [{"query": "grocery shopping", "n_results": 15}]
        test_case = {"expected_search_terms": ["restaurants", "grocery"]}

        score, detail = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        assert score == 1.0


class TestRetrievalEvaluatorIntegration:
    """Integration tests combining usage and relevance evaluation."""

    def test_full_evaluation_flow_success(
        self, retrieval_evaluator, test_case_needs_vector
    ):
        """Full evaluation should succeed with proper vector search."""
        vector_calls = [{"query": "restaurants", "n_results": 15}]

        usage_score, _ = retrieval_evaluator.evaluate_vector_search_usage(
            vector_calls, test_case_needs_vector
        )
        relevance_score, _ = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case_needs_vector
        )

        assert usage_score == 1.0
        assert relevance_score == 1.0

    def test_full_evaluation_flow_no_vector_needed(
        self, retrieval_evaluator, test_case_no_vector
    ):
        """Full evaluation should succeed when vector search not needed."""
        vector_calls = []

        usage_score, _ = retrieval_evaluator.evaluate_vector_search_usage(
            vector_calls, test_case_no_vector
        )
        relevance_score, _ = retrieval_evaluator.evaluate_retrieval_relevance(
            vector_calls, test_case_no_vector
        )

        assert usage_score == 1.0
        assert relevance_score == 1.0

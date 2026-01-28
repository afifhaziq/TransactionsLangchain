"""
Tests for the Tier 3 Response Evaluator.

These tests validate response faithfulness checking,
hallucination detection, and amount accuracy verification.
"""

from evaluation.tier3_response import ResponseEvaluator


class TestFaithfulness:
    """Test hallucination and placeholder detection."""

    def test_passes_clean_response(self, response_evaluator):
        """Clean responses without placeholders should pass."""
        response = "You spent $500.00 on groceries in August."
        score, detail = response_evaluator.evaluate_faithfulness(
            response, golden_output=[], transaction_ids=[]
        )
        assert score == 1.0
        assert "faithful" in detail.lower()

    def test_detects_placeholder_amount(self, response_evaluator):
        """Should detect $X.XX placeholder pattern."""
        response = "Your total spending was $X.XX"
        score, detail = response_evaluator.evaluate_faithfulness(
            response, golden_output=[], transaction_ids=[]
        )
        assert score == 0.0
        assert "placeholder" in detail.lower()

    def test_detects_placeholder_text_actual(self, response_evaluator):
        """Should detect [actual...] placeholder pattern."""
        response = "The date was [actual date here]"
        score, detail = response_evaluator.evaluate_faithfulness(
            response, golden_output=[], transaction_ids=[]
        )
        assert score == 0.0
        assert "placeholder" in detail.lower()

    def test_detects_placeholder_text_transaction(self, response_evaluator):
        """Should detect [transaction] placeholder pattern."""
        response = "Here are your [transaction] details"
        score, detail = response_evaluator.evaluate_faithfulness(
            response, golden_output=[], transaction_ids=[]
        )
        assert score == 0.0
        assert "placeholder" in detail.lower()

    def test_detects_fabricated_sequential_ids(self, response_evaluator):
        """Should detect fabricated sequential transaction IDs."""
        response = "Transactions: 1, 2, 3, 4, 5"
        score, detail = response_evaluator.evaluate_faithfulness(
            response, golden_output=[], transaction_ids=[1, 2, 3, 4, 5]
        )
        assert score == 0.0
        assert "sequential" in detail.lower() or "fabricated" in detail.lower()

    def test_detects_hallucinated_transaction_ids(
        self, response_evaluator, golden_output_with_transactions
    ):
        """Should detect transaction IDs not in golden output."""
        response = "Transaction ID: 999999"  # Not in golden output
        score, detail = response_evaluator.evaluate_faithfulness(
            response,
            golden_output=golden_output_with_transactions,
            transaction_ids=[999999],
        )
        assert score == 0.0
        assert "hallucinated" in detail.lower()

    def test_passes_valid_transaction_ids(
        self, response_evaluator, golden_output_with_transactions
    ):
        """Should pass when transaction IDs match golden output."""
        response = "Transaction ID: 114224"  # Exists in golden output
        score, detail = response_evaluator.evaluate_faithfulness(
            response,
            golden_output=golden_output_with_transactions,
            transaction_ids=[114224],
        )
        assert score == 1.0


class TestSequentialDetection:
    """Test the _is_sequential helper function."""

    def test_detects_sequential_from_one(self):
        """Should detect 1,2,3,4,5 as sequential."""
        assert ResponseEvaluator._is_sequential([1, 2, 3, 4, 5]) is True

    def test_detects_sequential_in_larger_set(self):
        """Should detect sequential subset in larger list."""
        assert ResponseEvaluator._is_sequential([100, 5, 6, 7, 200]) is True

    def test_non_sequential_returns_false(self):
        """Non-sequential IDs should return False."""
        assert ResponseEvaluator._is_sequential([114224, 46120, 156183]) is False

    def test_short_list_returns_false(self):
        """Lists with fewer than 3 items should return False."""
        assert ResponseEvaluator._is_sequential([1, 2]) is False

    def test_empty_list_returns_false(self):
        """Empty list should return False."""
        assert ResponseEvaluator._is_sequential([]) is False

    def test_single_item_returns_false(self):
        """Single item list should return False."""
        assert ResponseEvaluator._is_sequential([42]) is False

    def test_detects_sequential_at_end(self):
        """Should detect sequential pattern at end of list."""
        assert ResponseEvaluator._is_sequential([1000, 2000, 8, 9, 10]) is True

    def test_large_sequential_gap(self):
        """Large gaps between numbers should not be sequential."""
        assert ResponseEvaluator._is_sequential([1, 100, 200, 300, 400]) is False


class TestAmountAccuracy:
    """Test amount validation in responses."""

    def test_passes_matching_amount(self, response_evaluator):
        """Should pass when amount matches expected."""
        test_case = {"expected_amount": 500.0}
        score, detail = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[500.0],
            test_case=test_case,
            golden_output=[{"amt": 500.0}],
        )
        assert score == 1.0
        assert "accurate" in detail.lower()

    def test_passes_within_tolerance(self, response_evaluator):
        """Should pass when amount within $0.01 tolerance."""
        test_case = {"expected_amount": 500.0}
        score, detail = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[500.01],  # Within tolerance
            test_case=test_case,
            golden_output=[{"amt": 500.0}],
        )
        assert score == 1.0

    def test_fails_wrong_amount(self, response_evaluator):
        """Should fail when amount doesn't match."""
        test_case = {"expected_amount": 500.0}
        score, detail = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[600.0],
            test_case=test_case,
            golden_output=[{"amt": 500.0}],
        )
        assert score == 0.0
        assert "mismatch" in detail.lower()

    def test_handles_expected_spending(self, response_evaluator):
        """Should validate expected_spending field."""
        test_case = {"expected_spending": -602027.27}
        score, detail = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[-602027.27],
            test_case=test_case,
            golden_output=[{"total_spending": -602027.27}],
        )
        assert score == 1.0

    def test_handles_expected_income(self, response_evaluator):
        """Should validate expected_income field."""
        test_case = {"expected_income": 776.68}
        score, detail = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[776.68],
            test_case=test_case,
            golden_output=[{"total_income": 776.68}],
        )
        assert score == 1.0

    def test_handles_empty_response_amounts(self, response_evaluator):
        """Should handle no amounts in response."""
        test_case = {"expected_amount": 500.0}
        score, detail = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[],
            test_case=test_case,
            golden_output=[{"amt": 500.0}],
        )
        assert score == 0.0
        assert "no amounts" in detail.lower()

    def test_handles_empty_result_correctly(self, response_evaluator):
        """Should handle expected empty result."""
        test_case = {}  # No expected amounts
        score, detail = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[],
            test_case=test_case,
            golden_output=[],  # Empty golden output
        )
        assert score == 1.0
        assert "empty result" in detail.lower()

    def test_finds_closest_amount(self, response_evaluator):
        """Should find closest matching amount in list."""
        test_case = {"expected_amount": 500.0}
        score, detail = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[100.0, 500.0, 900.0],  # 500.0 is exact match
            test_case=test_case,
            golden_output=[{"amt": 500.0}],
        )
        assert score == 1.0

    def test_negative_amounts(self, response_evaluator):
        """Should handle negative amounts (spending)."""
        test_case = {"expected_amount": -638532.93}
        score, detail = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[-638532.93],
            test_case=test_case,
            golden_output=[{"total_spending": -638532.93}],
        )
        assert score == 1.0


class TestResponseEvaluatorIntegration:
    """Integration tests for response evaluation."""

    def test_complete_response_evaluation(
        self, response_evaluator, golden_output_with_transactions
    ):
        """Test complete response evaluation flow."""
        response = "Transaction ID: 114224, Amount: $-2.22"

        faith_score, _ = response_evaluator.evaluate_faithfulness(
            response,
            golden_output=golden_output_with_transactions,
            transaction_ids=[114224],
        )

        test_case = {"expected_amount": -2.22}
        amount_score, _ = response_evaluator.evaluate_amount_accuracy(
            response_amounts=[-2.22],
            test_case=test_case,
            golden_output=golden_output_with_transactions,
        )

        assert faith_score == 1.0
        assert amount_score == 1.0

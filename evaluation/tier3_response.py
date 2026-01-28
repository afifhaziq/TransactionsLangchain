"""
Tier 3: Response Quality Evaluators.

Validates response faithfulness, formatting, and accuracy.
"""

from typing import Tuple, List, Dict


class ResponseEvaluator:
    """Evaluates quality of agent responses."""

    @staticmethod
    def evaluate_faithfulness(
        response: str, golden_output: List[Dict], transaction_ids: List[int]
    ) -> Tuple[float, str]:
        """
        Verify response is faithful to actual database data.

        Checks for:
        - Placeholder text
        - Fabricated sequential IDs
        - Hallucinated transaction IDs
        """
        issues = []

        # Check for placeholder text
        placeholders = ["$X.XX", "$Y.YY", "[actual", "[transaction", "placeholder"]
        for placeholder in placeholders:
            if placeholder.lower() in response.lower():
                issues.append(f"Contains placeholder: {placeholder}")

        # Check for fabricated sequential IDs (1,2,3,4,5)
        if transaction_ids and ResponseEvaluator._is_sequential(transaction_ids):
            issues.append("CRITICAL: Sequential fabricated IDs detected")

        # Verify transaction IDs against golden output
        if golden_output and len(golden_output) > 0:
            if "txn_id" in golden_output[0]:
                expected_ids = {row["txn_id"] for row in golden_output}
                hallucinated = [
                    tid for tid in transaction_ids if tid not in expected_ids
                ]
                if hallucinated:
                    issues.append(f"Hallucinated transaction IDs: {hallucinated}")

        if not issues:
            return 1.0, "Response is faithful to DB results"
        else:
            return 0.0, f"Faithfulness violations: {'; '.join(issues)}"

    @staticmethod
    def evaluate_amount_accuracy(
        response_amounts: List[float], test_case: Dict, golden_output: List[Dict]
    ) -> Tuple[float, str]:
        """
        Verify amounts in response match expected values.

        Checks against expected_amount, expected_spending, expected_income from test case.
        """
        # Handle empty results
        if not response_amounts:
            if len(golden_output) == 0 or (
                len(golden_output) == 1
                and all(v is None or v == 0 for v in golden_output[0].values())
            ):
                return 1.0, "Correctly handled empty result"
            else:
                return 0.0, "No amounts found in response"

        issues = []

        # Check expected amount
        expected_amount = test_case.get("expected_amount")
        if expected_amount is not None:
            closest = min(response_amounts, key=lambda x: abs(x - expected_amount))
            if abs(closest - expected_amount) > 0.01:
                issues.append(
                    f"Amount mismatch: expected ${expected_amount:.2f}, "
                    f"closest found ${closest:.2f}"
                )

        # Check expected spending
        expected_spending = test_case.get("expected_spending")
        if expected_spending is not None:
            closest = min(response_amounts, key=lambda x: abs(x - expected_spending))
            if abs(closest - expected_spending) > 0.01:
                issues.append(
                    f"Spending mismatch: expected ${expected_spending:.2f}, "
                    f"closest found ${closest:.2f}"
                )

        # Check expected income
        expected_income = test_case.get("expected_income")
        if expected_income is not None:
            closest = min(response_amounts, key=lambda x: abs(x - expected_income))
            if abs(closest - expected_income) > 0.01:
                issues.append(
                    f"Income mismatch: expected ${expected_income:.2f}, "
                    f"found ${closest:.2f}"
                )

        if not issues:
            return 1.0, "Amounts are accurate"
        else:
            return 0.0, f"{'; '.join(issues)}"

    @staticmethod
    def _is_sequential(numbers: List[int]) -> bool:
        """Check if numbers are sequential (e.g., 1,2,3,4,5)."""
        if len(numbers) < 3:
            return False

        sorted_nums = sorted(numbers)
        for i in range(len(sorted_nums) - 2):
            if (
                sorted_nums[i + 1] == sorted_nums[i] + 1
                and sorted_nums[i + 2] == sorted_nums[i] + 2
            ):
                return True

        return False

"""
Tests for the Tier 1 Functional Evaluator.

These tests validate SQL security compliance, result comparison,
and the detection of SQL injection patterns and dangerous operations.
"""

import pytest

from .conftest import (
    SECURITY_TEST_CASES_VALID,
    SECURITY_TEST_CASES_INJECTION,
    SECURITY_TEST_CASES_DANGEROUS,
    SECURITY_TEST_CASES_MISSING_CLIENT,
)


class TestSecurityComplianceValid:
    """Test that valid SQL queries pass security checks."""

    @pytest.mark.parametrize(
        "test_case",
        SECURITY_TEST_CASES_VALID,
        ids=[tc["name"] for tc in SECURITY_TEST_CASES_VALID],
    )
    def test_valid_queries_pass(self, functional_evaluator, test_case):
        """Valid queries with proper client_id should pass."""
        score, detail = functional_evaluator.evaluate_security_compliance(
            test_case["sql"], test_case["client_id"]
        )
        assert score == test_case["expected_score"], (
            f"Failed: {test_case['name']}, detail: {detail}"
        )
        assert "passed" in detail.lower(), f"Expected 'passed' in detail: {detail}"


class TestSecurityComplianceInjection:
    """Test that SQL injection patterns are blocked."""

    @pytest.mark.parametrize(
        "test_case",
        SECURITY_TEST_CASES_INJECTION,
        ids=[tc["name"] for tc in SECURITY_TEST_CASES_INJECTION],
    )
    def test_injection_patterns_blocked(self, functional_evaluator, test_case):
        """SQL injection patterns should be detected and blocked."""
        score, detail = functional_evaluator.evaluate_security_compliance(
            test_case["sql"], test_case["client_id"]
        )
        assert score == test_case["expected_score"], (
            f"Failed to block: {test_case['name']}"
        )
        assert test_case["detail_contains"].lower() in detail.lower(), (
            f"Expected '{test_case['detail_contains']}' in detail: {detail}"
        )


class TestSecurityComplianceDangerous:
    """Test that dangerous SQL operations are blocked."""

    @pytest.mark.parametrize(
        "test_case",
        SECURITY_TEST_CASES_DANGEROUS,
        ids=[tc["name"] for tc in SECURITY_TEST_CASES_DANGEROUS],
    )
    def test_dangerous_operations_blocked(self, functional_evaluator, test_case):
        """Dangerous operations (DROP, DELETE, etc.) should be blocked."""
        score, detail = functional_evaluator.evaluate_security_compliance(
            test_case["sql"], test_case["client_id"]
        )
        assert score == test_case["expected_score"], (
            f"Failed to block: {test_case['name']}"
        )
        assert test_case["detail_contains"].lower() in detail.lower(), (
            f"Expected '{test_case['detail_contains']}' in detail: {detail}"
        )


class TestSecurityComplianceMissingClient:
    """Test that queries missing client_id are blocked."""

    @pytest.mark.parametrize(
        "test_case",
        SECURITY_TEST_CASES_MISSING_CLIENT,
        ids=[tc["name"] for tc in SECURITY_TEST_CASES_MISSING_CLIENT],
    )
    def test_missing_client_id_blocked(self, functional_evaluator, test_case):
        """Queries without proper client_id filtering should be blocked."""
        score, detail = functional_evaluator.evaluate_security_compliance(
            test_case["sql"], test_case["client_id"]
        )
        assert score == test_case["expected_score"], (
            f"Failed to block: {test_case['name']}"
        )
        # Detail should mention missing or incorrect client_id
        assert any(word in detail.lower() for word in ["missing", "incorrect"]), (
            f"Expected 'missing' or 'incorrect' in detail: {detail}"
        )


class TestResultComparison:
    """Test SQL result set comparison logic."""

    def test_identical_results_match(self, functional_evaluator):
        """Identical result sets should match."""
        result1 = [{"amt": 100.0, "cat": "Food"}]
        result2 = [{"amt": 100.0, "cat": "Food"}]
        assert functional_evaluator._compare_results(result1, result2) is True

    def test_different_values_dont_match(self, functional_evaluator):
        """Different values should not match."""
        result1 = [{"amt": 100.0}]
        result2 = [{"amt": 200.0}]
        assert functional_evaluator._compare_results(result1, result2) is False

    def test_float_rounding_tolerance(self, functional_evaluator):
        """Floats within rounding tolerance should match."""
        result1 = [{"amt": 100.004}]
        result2 = [{"amt": 100.001}]
        # Both round to 100.00 with 2 decimal places
        assert functional_evaluator._compare_results(result1, result2) is True

    def test_different_column_names_same_values(self, functional_evaluator):
        """Different column aliases with same values should match."""
        result1 = [{"total_spent": -500.0}]
        result2 = [{"total_spending": -500.0}]
        assert functional_evaluator._compare_results(result1, result2) is True

    def test_empty_results_match(self, functional_evaluator):
        """Empty result sets should match."""
        assert functional_evaluator._compare_results([], []) is True

    def test_different_lengths_dont_match(self, functional_evaluator):
        """Result sets of different lengths should not match."""
        result1 = [{"amt": 100.0}]
        result2 = [{"amt": 100.0}, {"amt": 200.0}]
        assert functional_evaluator._compare_results(result1, result2) is False

    def test_multiple_rows_match(self, functional_evaluator):
        """Multiple rows with same values should match."""
        result1 = [{"amt": 100.0}, {"amt": 200.0}]
        result2 = [{"amt": 100.0}, {"amt": 200.0}]
        assert functional_evaluator._compare_results(result1, result2) is True

    def test_unordered_results_match(self, functional_evaluator):
        """Results in different order should still match."""
        result1 = [{"amt": 200.0}, {"amt": 100.0}]
        result2 = [{"amt": 100.0}, {"amt": 200.0}]
        assert functional_evaluator._compare_results(result1, result2) is True


class TestAdditionalSecurityPatterns:
    """Test additional SQL injection and security patterns."""

    def test_multi_line_comment_injection(self, functional_evaluator):
        """Multi-line comment injection should be blocked."""
        sql = "SELECT * FROM t WHERE clnt_id = 880 /* admin bypass */"
        score, detail = functional_evaluator.evaluate_security_compliance(sql, 880)
        assert score == 0.0
        assert "injection" in detail.lower()

    def test_semicolon_stacked_query(self, functional_evaluator):
        """Stacked queries with semicolon should be blocked."""
        sql = "SELECT * FROM t WHERE clnt_id = 880; DROP TABLE users"
        score, detail = functional_evaluator.evaluate_security_compliance(sql, 880)
        assert score == 0.0

    def test_client_id_in_subquery(self, functional_evaluator):
        """Client ID in proper WHERE clause should pass."""
        sql = "SELECT * FROM transactions WHERE clnt_id = 880 AND amt IN (SELECT amt FROM other)"
        score, detail = functional_evaluator.evaluate_security_compliance(sql, 880)
        # Should pass because clnt_id = 880 is present
        assert score == 1.0

    def test_alter_table_blocked(self, functional_evaluator):
        """ALTER TABLE should be blocked."""
        sql = "ALTER TABLE transactions ADD COLUMN hack VARCHAR(100)"
        score, detail = functional_evaluator.evaluate_security_compliance(sql, 880)
        assert score == 0.0
        assert "dangerous" in detail.lower()

    def test_create_table_blocked(self, functional_evaluator):
        """CREATE TABLE should be blocked."""
        sql = "CREATE TABLE hacked (id INT)"
        score, detail = functional_evaluator.evaluate_security_compliance(sql, 880)
        assert score == 0.0
        assert "dangerous" in detail.lower()

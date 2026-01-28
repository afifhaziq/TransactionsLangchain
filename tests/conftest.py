"""
Shared pytest fixtures for the RAG-SQL Agent test suite.

These fixtures provide common test data and mock objects that don't
require external dependencies like Ollama or real transaction data.
"""

import pytest


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary empty SQLite database file."""
    db_path = tmp_path / "test.db"
    db_path.touch()
    return str(db_path)


@pytest.fixture
def functional_evaluator(temp_db_path):
    """Create a FunctionalEvaluator with a temporary database."""
    from evaluation.tier1_functional import FunctionalEvaluator

    evaluator = FunctionalEvaluator(temp_db_path)
    yield evaluator
    evaluator.close()


@pytest.fixture
def retrieval_evaluator():
    """Create a RetrievalEvaluator instance."""
    from evaluation.tier2_retrieval import RetrievalEvaluator

    return RetrievalEvaluator()


@pytest.fixture
def response_evaluator():
    """Create a ResponseEvaluator instance."""
    from evaluation.tier3_response import ResponseEvaluator

    return ResponseEvaluator()


# ============================================================
# Security Test Cases Data
# ============================================================

SECURITY_TEST_CASES_VALID = [
    {
        "name": "simple_select_with_client_id",
        "sql": "SELECT * FROM transactions WHERE clnt_id = 880",
        "client_id": 880,
        "expected_score": 1.0,
    },
    {
        "name": "aggregation_with_client_id",
        "sql": "SELECT SUM(amt) FROM transactions WHERE clnt_id = 880 AND amt < 0",
        "client_id": 880,
        "expected_score": 1.0,
    },
    {
        "name": "select_with_date_filter",
        "sql": "SELECT * FROM transactions WHERE clnt_id = 880 AND txn_date LIKE '%/08/2023%'",
        "client_id": 880,
        "expected_score": 1.0,
    },
    {
        "name": "select_with_category_filter",
        "sql": "SELECT * FROM transactions WHERE clnt_id = 880 AND cat = 'Restaurants'",
        "client_id": 880,
        "expected_score": 1.0,
    },
]

SECURITY_TEST_CASES_INJECTION = [
    {
        "name": "or_1_equals_1",
        "sql": "SELECT * FROM transactions WHERE clnt_id = 880 OR 1=1",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "injection",
    },
    {
        "name": "union_select",
        "sql": "SELECT * FROM transactions WHERE clnt_id = 880 UNION SELECT * FROM users",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "injection",
    },
    {
        "name": "comment_injection",
        "sql": "SELECT * FROM transactions WHERE clnt_id = 880 -- admin bypass",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "injection",
    },
    {
        "name": "or_string_equals_string",
        "sql": "SELECT * FROM transactions WHERE clnt_id = 880 OR 'a'='a'",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "injection",
    },
]

SECURITY_TEST_CASES_DANGEROUS = [
    {
        "name": "drop_table",
        "sql": "DROP TABLE transactions",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "dangerous",
    },
    {
        "name": "delete_from",
        "sql": "DELETE FROM transactions WHERE clnt_id = 880",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "dangerous",
    },
    {
        "name": "update_table",
        "sql": "UPDATE transactions SET amt = 0 WHERE clnt_id = 880",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "dangerous",
    },
    {
        "name": "insert_into",
        "sql": "INSERT INTO transactions (clnt_id, amt) VALUES (880, 1000)",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "dangerous",
    },
    {
        "name": "truncate_table",
        "sql": "TRUNCATE TABLE transactions",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "dangerous",
    },
]

SECURITY_TEST_CASES_MISSING_CLIENT = [
    {
        "name": "missing_client_id",
        "sql": "SELECT * FROM transactions WHERE amt < 0",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "missing",
    },
    {
        "name": "wrong_client_id",
        "sql": "SELECT * FROM transactions WHERE clnt_id = 999",
        "client_id": 880,
        "expected_score": 0.0,
        "detail_contains": "missing",
    },
]


@pytest.fixture
def valid_security_cases():
    """Valid SQL queries that should pass security checks."""
    return SECURITY_TEST_CASES_VALID


@pytest.fixture
def injection_security_cases():
    """SQL injection patterns that should be blocked."""
    return SECURITY_TEST_CASES_INJECTION


@pytest.fixture
def dangerous_security_cases():
    """Dangerous SQL operations that should be blocked."""
    return SECURITY_TEST_CASES_DANGEROUS


@pytest.fixture
def missing_client_cases():
    """Queries missing proper client_id filtering."""
    return SECURITY_TEST_CASES_MISSING_CLIENT


# ============================================================
# Mock Agent Output Data
# ============================================================


@pytest.fixture
def mock_agent_output_with_sql():
    """Mock agent output containing SQL tool call."""

    class MockMessage:
        def __init__(self):
            self.tool_calls = [
                {
                    "name": "sql_db_query",
                    "args": {
                        "query": "SELECT SUM(amt) FROM transactions WHERE clnt_id = 880 AND amt < 0"
                    },
                }
            ]
            self.content = ""

    return {"messages": [MockMessage()]}


@pytest.fixture
def mock_agent_output_with_vector_search():
    """Mock agent output containing vector search tool call."""

    class MockMessage:
        def __init__(self):
            self.tool_calls = [
                {
                    "name": "vector_search",
                    "args": {"query": "restaurants", "n_results": 15},
                }
            ]
            self.content = ""

    return {"messages": [MockMessage()]}


@pytest.fixture
def mock_agent_output_empty():
    """Mock agent output with no tool calls."""
    return {"messages": []}


# ============================================================
# Test Case Data for Retrieval Evaluation
# ============================================================


@pytest.fixture
def test_case_needs_vector():
    """Test case that requires vector search."""
    return {
        "test_id": "TC002",
        "question": "How much did I spend on restaurants?",
        "need_vector": True,
        "expected_search_terms": ["restaurants"],
    }


@pytest.fixture
def test_case_no_vector():
    """Test case that doesn't require vector search."""
    return {
        "test_id": "TC001",
        "question": "How much did I spend in August 2023?",
        "need_vector": False,
        "expected_search_terms": [],
    }


# ============================================================
# Golden Output Data for Response Evaluation
# ============================================================


@pytest.fixture
def golden_output_with_transactions():
    """Golden output containing transaction data."""
    return [
        {
            "txn_id": 114224,
            "txn_date": "01/06/2023 0:00",
            "desc": "McDonald's",
            "merchant": "MCDONALD'S",
            "cat": "Restaurants",
            "amt": -2.22,
        },
        {
            "txn_id": 46120,
            "txn_date": "01/06/2023 0:00",
            "desc": "Debit Purchase",
            "merchant": None,
            "cat": "Restaurants",
            "amt": -3.29,
        },
    ]


@pytest.fixture
def golden_output_empty():
    """Golden output for queries with no results."""
    return []


@pytest.fixture
def golden_output_aggregation():
    """Golden output for aggregation queries."""
    return [{"total_spending": -638532.93}]

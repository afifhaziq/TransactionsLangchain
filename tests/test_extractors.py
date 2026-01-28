"""
Tests for the AgentOutputExtractor class.

These tests validate the extraction of SQL queries, vector search calls,
monetary amounts, and transaction IDs from agent output structures.
"""

from evaluation.extractors import AgentOutputExtractor


class TestAmountExtraction:
    """Test monetary amount parsing from response text."""

    def test_extracts_simple_amount(self):
        """Should extract a simple dollar amount."""
        amounts = AgentOutputExtractor.extract_amounts("Total: $500.00")
        assert amounts == [500.0]

    def test_extracts_amount_with_commas(self):
        """Should extract amounts with thousand separators."""
        amounts = AgentOutputExtractor.extract_amounts(
            "You spent $1,234.56 on groceries"
        )
        assert amounts == [1234.56]

    def test_extracts_negative_amount(self):
        """Should extract negative amounts."""
        amounts = AgentOutputExtractor.extract_amounts("Balance: $-100.50")
        assert amounts == [-100.5]

    def test_extracts_multiple_amounts(self):
        """Should extract all amounts from text with multiple values."""
        text = "Income: $500.00, Spending: $-300.00, Net: $200.00"
        amounts = AgentOutputExtractor.extract_amounts(text)
        assert amounts == [500.0, -300.0, 200.0]

    def test_extracts_large_amount(self):
        """Should extract large amounts with multiple comma separators."""
        amounts = AgentOutputExtractor.extract_amounts("Total: $2,472,806.78")
        assert amounts == [2472806.78]

    def test_returns_empty_for_no_amounts(self):
        """Should return empty list when no amounts present."""
        amounts = AgentOutputExtractor.extract_amounts("No transactions found")
        assert amounts == []

    def test_extracts_amount_without_cents(self):
        """Should extract whole dollar amounts."""
        amounts = AgentOutputExtractor.extract_amounts("Total: $500")
        assert amounts == [500.0]

    def test_handles_amount_in_sentence(self):
        """Should extract amounts embedded in sentences."""
        text = "You spent $638,532.93 in August 2023 on various purchases."
        amounts = AgentOutputExtractor.extract_amounts(text)
        assert amounts == [638532.93]


class TestTransactionIdExtraction:
    """Test transaction ID parsing from response text."""

    def test_extracts_transaction_id_format(self):
        """Should extract 'Transaction ID: X' format."""
        text = "Transaction ID: 12345"
        ids = AgentOutputExtractor.extract_transaction_ids(text)
        assert 12345 in ids

    def test_extracts_txn_id_format(self):
        """Should extract 'txn_id: X' format."""
        text = "txn_id: 67890"
        ids = AgentOutputExtractor.extract_transaction_ids(text)
        assert 67890 in ids

    def test_extracts_multiple_ids(self):
        """Should extract multiple transaction IDs."""
        text = """
        Transaction ID: 114224
        Transaction ID: 46120
        Transaction ID: 156183
        """
        ids = AgentOutputExtractor.extract_transaction_ids(text)
        assert set(ids) == {114224, 46120, 156183}

    def test_returns_empty_for_no_ids(self):
        """Should return empty list when no IDs present."""
        ids = AgentOutputExtractor.extract_transaction_ids("No transactions found")
        assert ids == []

    def test_extracts_mixed_formats(self):
        """Should extract IDs in both formats."""
        text = "Transaction ID: 111, also txn_id: 222"
        ids = AgentOutputExtractor.extract_transaction_ids(text)
        assert set(ids) == {111, 222}

    def test_case_insensitive_transaction_id(self):
        """Should handle case variations in 'Transaction ID'."""
        text = "transaction ID: 12345"
        ids = AgentOutputExtractor.extract_transaction_ids(text)
        assert 12345 in ids


class TestSqlExtraction:
    """Test SQL query extraction from agent outputs."""

    def test_extracts_from_tool_calls(self, mock_agent_output_with_sql):
        """Should extract SQL from structured tool_calls."""
        sql = AgentOutputExtractor.extract_sql(mock_agent_output_with_sql)
        assert "SELECT" in sql
        assert "clnt_id = 880" in sql

    def test_returns_empty_for_no_sql(self, mock_agent_output_empty):
        """Should return empty string when no SQL present."""
        sql = AgentOutputExtractor.extract_sql(mock_agent_output_empty)
        assert sql == ""

    def test_returns_empty_for_invalid_input(self):
        """Should handle invalid input gracefully."""
        # None input
        sql = AgentOutputExtractor.extract_sql(None)
        assert sql == ""

        # Empty dict
        sql = AgentOutputExtractor.extract_sql({})
        assert sql == ""

        # Empty messages list
        sql = AgentOutputExtractor.extract_sql({"messages": []})
        assert sql == ""

    def test_extracts_from_dict_tool_calls(self):
        """Should extract SQL from dict-based tool calls."""
        result = {
            "messages": [
                {
                    "tool_calls": [
                        {
                            "name": "sql_db_query",
                            "args": {
                                "query": "SELECT * FROM transactions WHERE clnt_id = 880"
                            },
                        }
                    ],
                    "content": "",
                }
            ]
        }
        sql = AgentOutputExtractor.extract_sql(result)
        assert "SELECT" in sql


class TestVectorSearchExtraction:
    """Test vector search call extraction from agent outputs."""

    def test_extracts_vector_search_calls(self, mock_agent_output_with_vector_search):
        """Should extract vector search calls from tool_calls."""
        calls = AgentOutputExtractor.extract_vector_search_calls(
            mock_agent_output_with_vector_search
        )
        assert len(calls) == 1
        assert calls[0]["query"] == "restaurants"
        assert calls[0]["n_results"] == 15

    def test_returns_empty_for_no_vector_calls(self, mock_agent_output_with_sql):
        """Should return empty list when no vector search calls."""
        calls = AgentOutputExtractor.extract_vector_search_calls(
            mock_agent_output_with_sql
        )
        assert calls == []

    def test_returns_empty_for_empty_output(self, mock_agent_output_empty):
        """Should return empty list for empty output."""
        calls = AgentOutputExtractor.extract_vector_search_calls(
            mock_agent_output_empty
        )
        assert calls == []

    def test_extracts_multiple_vector_calls(self):
        """Should extract multiple vector search calls."""

        class MockMessage:
            def __init__(self):
                self.tool_calls = [
                    {
                        "name": "vector_search",
                        "args": {"query": "restaurants", "n_results": 15},
                    },
                    {
                        "name": "vector_search",
                        "args": {"query": "groceries", "n_results": 10},
                    },
                ]
                self.content = ""

        result = {"messages": [MockMessage()]}
        calls = AgentOutputExtractor.extract_vector_search_calls(result)

        assert len(calls) == 2
        assert calls[0]["query"] == "restaurants"
        assert calls[1]["query"] == "groceries"

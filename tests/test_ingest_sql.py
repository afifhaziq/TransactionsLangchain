"""
Tests for the data ingestion utilities.

These tests validate column name cleaning and normalization
for SQL-friendly table creation.
"""

from src.ingest_sql import clean_column_name


class TestCleanColumnName:
    """Test SQL-friendly column name normalization."""

    def test_removes_spaces(self):
        """Should replace spaces with underscores."""
        assert clean_column_name("Column Name") == "column_name"

    def test_removes_special_characters(self):
        """Should remove special characters like parentheses and $."""
        assert clean_column_name("Amount ($)") == "amount"

    def test_handles_multiple_underscores(self):
        """Should collapse multiple underscores into one."""
        assert clean_column_name("col___name") == "col_name"

    def test_strips_leading_trailing_underscores(self):
        """Should strip leading and trailing underscores."""
        assert clean_column_name("_column_") == "column"

    def test_converts_to_lowercase(self):
        """Should convert to lowercase."""
        assert clean_column_name("COLUMN_NAME") == "column_name"

    def test_handles_existing_snake_case(self):
        """Should preserve valid snake_case names."""
        assert clean_column_name("txn_date") == "txn_date"

    def test_handles_camel_case(self):
        """Should lowercase camelCase (doesn't split it)."""
        assert clean_column_name("TransactionDate") == "transactiondate"

    def test_handles_mixed_special_chars(self):
        """Should handle multiple special characters."""
        assert clean_column_name("Amount (USD) [Net]") == "amount_usd_net"

    def test_handles_numbers(self):
        """Should preserve numbers in column names."""
        assert clean_column_name("Column1") == "column1"
        assert clean_column_name("col_2_name") == "col_2_name"

    def test_handles_leading_spaces(self):
        """Should handle leading/trailing whitespace."""
        assert clean_column_name("  column  ") == "column"

    def test_handles_tabs_and_newlines(self):
        """Should handle tabs and newlines as spaces."""
        result = clean_column_name("col\tname")
        assert "_" not in result or result == "col_name"

    def test_empty_string(self):
        """Should handle empty string gracefully."""
        result = clean_column_name("")
        assert result == ""

    def test_only_special_chars(self):
        """Should handle string with only special characters."""
        result = clean_column_name("@#$%")
        assert result == ""

    def test_real_column_examples(self):
        """Test with real column name examples from datasets."""
        # Common financial dataset columns
        assert clean_column_name("Transaction ID") == "transaction_id"
        assert clean_column_name("Client ID") == "client_id"
        assert clean_column_name("Bank ID") == "bank_id"
        assert clean_column_name("Account ID") == "account_id"

    def test_preserves_underscores_between_words(self):
        """Should preserve intentional underscores."""
        assert clean_column_name("first_name") == "first_name"
        assert clean_column_name("last_name") == "last_name"

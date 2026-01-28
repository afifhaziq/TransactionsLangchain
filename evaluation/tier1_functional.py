"""
Tier 1: Functional Correctness Evaluators.

Validates SQL execution accuracy, validity, and security compliance.
"""

import sqlite3
import re
from typing import Tuple, List, Dict


class FunctionalEvaluator:
    """Evaluates functional correctness of SQL queries."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def execute_sql(self, sql: str) -> List[Dict]:
        """Execute SQL and return results as list of dicts."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            columns = (
                [desc[0] for desc in cursor.description] if cursor.description else []
            )
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            return {"error": str(e)}

    def evaluate_execution_accuracy(
        self, generated_sql: str, golden_sql: str
    ) -> Tuple[float, str]:
        """
        Compare results from generated SQL vs golden SQL.

        Returns:
            (score, detail_message) where score is 1.0 for exact match, 0.0 otherwise
        """
        try:
            gen_result = self.execute_sql(generated_sql)
            golden_result = self.execute_sql(golden_sql)

            if isinstance(gen_result, dict) and "error" in gen_result:
                return 0.0, f"Generated SQL failed: {gen_result['error']}"

            if self._compare_results(gen_result, golden_result):
                return 1.0, "Results match exactly"
            else:
                return 0.0, f"Mismatch - Expected: {golden_result}, Got: {gen_result}"

        except Exception as e:
            return 0.0, f"Execution error: {str(e)}"

    def evaluate_sql_validity(self, generated_sql: str) -> Tuple[float, str]:
        """Check if SQL executes without errors."""
        try:
            self.execute_sql(generated_sql)
            return 1.0, "SQL executed successfully"
        except Exception as e:
            return 0.0, f"SQL error: {str(e)}"

    def evaluate_security_compliance(
        self, generated_sql: str, expected_client_id: int
    ) -> Tuple[float, str]:
        """
        Verify client_id filtering is present and correct.
        Also checks for dangerous SQL operations and injection patterns.

        Uses regex to handle various SQL formatting styles.
        """
        # Check for dangerous SQL operations (data manipulation/destruction)
        dangerous_operations = [
            (r"\bINSERT\s+INTO\b", "INSERT"),
            (r"\bUPDATE\b", "UPDATE"),
            (r"\bDELETE\s+FROM\b", "DELETE"),
            (r"\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX)\b", "DROP"),
            (r"\bTRUNCATE\s+TABLE\b", "TRUNCATE"),
            (r"\bALTER\s+TABLE\b", "ALTER TABLE"),
            (r"\bCREATE\s+(TABLE|DATABASE|SCHEMA|INDEX)\b", "CREATE"),
            (r"\bEXEC(UTE)?\s*\(", "EXECUTE"),
            (r"\bGRANT\b", "GRANT"),
            (r"\bREVOKE\b", "REVOKE"),
        ]

        for pattern, operation in dangerous_operations:
            if re.search(pattern, generated_sql, re.IGNORECASE):
                return 0.0, f"CRITICAL: Dangerous operation detected ({operation})"

        # Check for common SQL injection patterns
        injection_patterns = [
            (r"--", "SQL comment injection (---)"),
            (r"/\*.*\*/", "Multi-line comment injection (/* */)"),
            (r";\s*DROP\b", "Statement chaining with DROP"),
            (r";\s*DELETE\b", "Statement chaining with DELETE"),
            (r"\bOR\s+1\s*=\s*1\b", "OR 1=1 injection"),
            (r"\bOR\s+'[^']*'\s*=\s*'[^']*'", "OR 'x'='x' injection"),
            (r"\bUNION\s+SELECT\b", "UNION SELECT injection"),
            (r"'\s*OR\s*'", "Quote-based OR injection"),
            (r"\bEXEC\s*\(", "Dynamic SQL execution"),
            (r"\bxp_cmdshell\b", "Command shell execution"),
        ]

        for pattern, description in injection_patterns:
            if re.search(pattern, generated_sql, re.IGNORECASE):
                return 0.0, f"CRITICAL: SQL injection pattern detected ({description})"

        # Check for proper client_id filtering
        # Pattern matches: clnt_id = 123, clnt_id=123, clnt_id IN (123), etc.
        client_id_patterns = [
            rf"clnt_id\s*=\s*{expected_client_id}\b",
            rf"clnt_id\s+IN\s*\(\s*{expected_client_id}\s*\)",
        ]

        for pattern in client_id_patterns:
            if re.search(pattern, generated_sql, re.IGNORECASE):
                return 1.0, f"Security check passed (clnt_id = {expected_client_id})"

        return 0.0, "CRITICAL: Missing or incorrect client_id filter"

    def _compare_results(self, result1: List[Dict], result2: List[Dict]) -> bool:
        """
        Compare two SQL result sets for equality.

        Handles cases where column aliases differ but values match.
        """
        if len(result1) != len(result2):
            return False

        # Normalize floats to 2 decimal places
        def normalize_row(row):
            return {
                k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
            }

        norm1 = [normalize_row(r) for r in result1]
        norm2 = [normalize_row(r) for r in result2]

        # Sort for comparison (order-independent)
        try:
            norm1_sorted = sorted(norm1, key=lambda x: str(sorted(x.items())))
            norm2_sorted = sorted(norm2, key=lambda x: str(sorted(x.items())))

            # First try exact match
            if norm1_sorted == norm2_sorted:
                return True

            # If keys differ but we have same number of rows, try value-only comparison
            # This handles cases where column aliases differ (e.g., total_spent vs total_spending)
            if len(norm1_sorted) == len(norm2_sorted):
                for row1, row2 in zip(norm1_sorted, norm2_sorted):
                    # Sort values for comparison
                    values1 = sorted(
                        row1.values(), key=lambda x: (type(x).__name__, str(x))
                    )
                    values2 = sorted(
                        row2.values(), key=lambda x: (type(x).__name__, str(x))
                    )
                    if values1 != values2:
                        return False
                return True

            return False
        except Exception:
            return False

    def close(self):
        """Close database connection."""
        self.conn.close()

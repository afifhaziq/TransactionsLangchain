"""
Extractors for parsing LangChain agent outputs.

Handles both structured tool_calls and JSON string formats.
"""

import json
import re
from typing import List, Dict, Any, Optional


class AgentOutputExtractor:
    """Extract structured information from LangChain agent outputs."""

    @staticmethod
    def extract_sql(result: Dict) -> str:
        """
        Extract SQL query from agent result.

        Handles two formats:
        1. Structured tool_calls: tool_calls=[{"name": "sql_db_query", "args": {...}}]
        2. JSON string content: content='{"tool": "sql_db_query", "query": "..."}'
        """
        if not isinstance(result, dict) or "messages" not in result:
            return ""

        for message in result["messages"]:
            # Format 1: Structured tool_calls (dict or attribute)
            tool_calls = getattr(message, "tool_calls", None) or (
                message.get("tool_calls") if isinstance(message, dict) else None
            )

            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.get("name") == "sql_db_query":
                        # Extract from args dict
                        args = tool_call.get("args", {})
                        sql = args.get("query", "").strip()
                        if sql.upper().startswith("SELECT"):
                            return sql

            # Format 2: JSON string in content
            content = getattr(message, "content", "") or (
                message.get("content", "") if isinstance(message, dict) else ""
            )

            if content and isinstance(content, str):
                sql = AgentOutputExtractor._extract_sql_from_json_string(content)
                if sql:
                    return sql

        return ""

    @staticmethod
    def _extract_sql_from_json_string(content: str) -> str:
        """Extract SQL from JSON string content."""
        content = content.strip()

        # Try parsing as JSON
        if content.startswith("{") and content.endswith("}"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # Check for sql_db_query tool
                    if (
                        data.get("tool") == "sql_db_query"
                        or data.get("name") == "sql_db_query"
                    ):
                        sql = data.get("query", "").strip()
                        if sql.upper().startswith("SELECT"):
                            return sql
            except json.JSONDecodeError:
                pass

        # Fallback: Direct SQL in content
        if content.upper().startswith("SELECT"):
            return content

        return ""

    @staticmethod
    def extract_vector_search_calls(result: Dict) -> List[Dict[str, Any]]:
        """
        Extract all vector_search tool calls from agent result.

        Returns list of dicts with 'query' and 'n_results' keys.
        """
        calls = []

        if not isinstance(result, dict) or "messages" not in result:
            return calls

        for message in result["messages"]:
            # Format 1: Structured tool_calls
            tool_calls = getattr(message, "tool_calls", None) or (
                message.get("tool_calls") if isinstance(message, dict) else None
            )

            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.get("name") == "vector_search":
                        args = tool_call.get("args", {})
                        calls.append(
                            {
                                "query": args.get("query", ""),
                                "n_results": args.get("n_results", 1),
                            }
                        )

            # Format 2: JSON string in content
            content = getattr(message, "content", "") or (
                message.get("content", "") if isinstance(message, dict) else ""
            )

            if content and isinstance(content, str):
                vector_call = AgentOutputExtractor._extract_vector_from_json_string(
                    content
                )
                if vector_call:
                    calls.append(vector_call)

        return calls

    @staticmethod
    def _extract_vector_from_json_string(content: str) -> Optional[Dict[str, Any]]:
        """Extract vector_search call from JSON string content."""
        content = content.strip()

        if content.startswith("{") and content.endswith("}"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    if (
                        data.get("tool") == "vector_search"
                        or data.get("name") == "vector_search"
                    ):
                        return {
                            "query": data.get("query", ""),
                            "n_results": data.get("n_results", 1),
                        }
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def extract_amounts(response: str) -> List[float]:
        """
        Extract monetary amounts from response text.

        Matches patterns like: $-1,234.56, $1,234.56, -$1234.56
        """
        pattern = r"\$(-?[0-9,]+\.?[0-9]*)"
        matches = re.findall(pattern, response)

        amounts = []
        for match in matches:
            try:
                amount = float(match.replace(",", ""))
                amounts.append(amount)
            except ValueError:
                continue

        return amounts

    @staticmethod
    def extract_transaction_ids(response: str) -> List[int]:
        """Extract transaction IDs from response text."""
        patterns = [
            r"[Tt]ransaction\s+ID:\s*(\d+)",
            r"txn_id:\s*(\d+)",
        ]

        ids = set()
        for pattern in patterns:
            matches = re.findall(pattern, response)
            ids.update(int(m) for m in matches)

        return sorted(ids)

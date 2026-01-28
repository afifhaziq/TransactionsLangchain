"""
Evaluation framework for SQL-RAG Agent.

This module provides a modular evaluation system with three tiers:
- Tier 1: Functional Correctness (SQL execution and security)
- Tier 2: Retrieval Precision (RAG quality)
- Tier 3: Response Quality (User experience)
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EvaluationResult:
    """Results from evaluating a single test case."""

    test_id: str
    question: str
    passed: bool
    tier1_score: float
    tier2_score: float
    tier3_score: float
    overall_score: float
    latency_seconds: float
    details: Dict[str, Any]


from .evaluator import AgentEvaluator  # noqa: E402

__all__ = ["EvaluationResult", "AgentEvaluator"]

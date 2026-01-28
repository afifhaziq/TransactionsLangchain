"""
Main evaluation orchestrator.

Coordinates all evaluation tiers and generates comprehensive reports.
"""

import time
import sys
import io
from typing import Dict, List

from . import EvaluationResult
from .extractors import AgentOutputExtractor
from .tier1_functional import FunctionalEvaluator
from .tier2_retrieval import RetrievalEvaluator
from .tier3_response import ResponseEvaluator


class AgentEvaluator:
    """
    Main evaluator that orchestrates all evaluation tiers.

    Usage:
        evaluator = AgentEvaluator(db_path="transactions.db")
        result = evaluator.evaluate_test_case(agent, test_case)
        evaluator.close()
    """

    def __init__(self, db_path: str = "transactions.db"):
        self.db_path = db_path
        self.functional = FunctionalEvaluator(db_path)
        self.retrieval = RetrievalEvaluator()
        self.response = ResponseEvaluator()
        self.extractor = AgentOutputExtractor()

    def evaluate_test_case(self, agent, test_case: Dict) -> EvaluationResult:
        """
        Evaluate a single test case.

        Args:
            agent: RagSqlAgent instance
            test_case: Dict with question, golden_sql, golden_output, etc.

        Returns:
            EvaluationResult with scores and details
        """

        # Ensure evaluations are single-turn and independent (no conversation bleed).
        if hasattr(agent, "reset_conversation"):
            try:
                agent.reset_conversation()
            except Exception:
                pass

        print(f"Evaluating {test_case['test_id']}: {test_case['question']}")

        # Run agent and measure latency
        response, result, latency = self._run_agent(agent, test_case["question"])
        print(f"Latency: {latency:.2f}s")
        print(f"\nAgent Response:\n{response[:300]}...")

        # Extract structured data
        generated_sql = self.extractor.extract_sql(result)
        vector_calls = self.extractor.extract_vector_search_calls(result)
        response_amounts = self.extractor.extract_amounts(response)
        transaction_ids = self.extractor.extract_transaction_ids(response)

        print(
            f"\nGenerated SQL:\n{generated_sql if generated_sql else 'No SQL extracted'}"
        )

        # Tier 1: Functional Correctness
        tier1_score, tier1_details = self._evaluate_tier1(generated_sql, test_case)

        # Tier 2: Retrieval Precision
        tier2_score, tier2_details = self._evaluate_tier2(vector_calls, test_case)

        # Tier 3: Response Quality
        tier3_score, tier3_details = self._evaluate_tier3(
            response, response_amounts, transaction_ids, test_case
        )

        # Calculate overall score (weighted scoring)
        overall_score = (
            tier1_score * 0.5  # Functional correctness
            + tier2_score * 0.25  # Retrieval quality
            + tier3_score * 0.25  # Response quality
        )

        passed = overall_score >= 0.8

        # Print summary
        self._print_summary(tier1_score, tier2_score, tier3_score, overall_score)

        return EvaluationResult(
            test_id=test_case["test_id"],
            question=test_case["question"],
            passed=passed,
            tier1_score=tier1_score,
            tier2_score=tier2_score,
            tier3_score=tier3_score,
            overall_score=overall_score,
            latency_seconds=latency,
            details={
                "generated_sql": generated_sql,
                "response": response,
                "tier1": tier1_details,
                "tier2": tier2_details,
                "tier3": tier3_details,
            },
        )

    def _run_agent(self, agent, question: str):
        """Run agent and capture response, result, and latency."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        start_time = time.time()
        try:
            # Use stream with invoke=True for synchronous execution
            result = agent.stream(question, invoke=True, remember=False)

            if not result or not isinstance(result, dict) or "messages" not in result:
                raise ValueError(f"Invalid result from agent: {type(result)}")

            response = getattr(result["messages"][-1], "content", "")
            # Avoid scoring on chain-of-thought if the model emits it.
            if hasattr(agent, "_strip_think"):
                try:
                    response = agent._strip_think(str(response))
                except Exception:
                    pass

        finally:
            sys.stdout = old_stdout
            latency = time.time() - start_time

        return response, result, latency

    def _evaluate_tier1(self, generated_sql: str, test_case: Dict):
        """Evaluate Tier 1: Functional Correctness."""
        print("\n--- Tier 1: Functional Correctness ---")

        if not generated_sql:
            # Check if this is an expected empty result case
            # If golden_output is empty and category is "empty_result",
            # not generating SQL is the correct behavior
            if test_case.get("golden_output") == []:
                print("Correctly handled empty result without SQL")
                return 1.0, {
                    "execution_accuracy": (
                        1.0,
                        "Correctly identified no data to query",
                    ),
                    "validity": (1.0, "No SQL needed for empty result"),
                    "security": (1.0, "No security risk (no SQL generated)"),
                }

            print("No SQL generated")
            return 0.0, {
                "execution_accuracy": (0.0, "No SQL generated"),
                "validity": (0.0, "No SQL to validate"),
                "security": (0.0, "No SQL to check"),
            }

        ex_score, ex_detail = self.functional.evaluate_execution_accuracy(
            generated_sql, test_case["golden_sql"]
        )
        val_score, val_detail = self.functional.evaluate_sql_validity(generated_sql)
        sec_score, sec_detail = self.functional.evaluate_security_compliance(
            generated_sql, test_case["client_id"]
        )

        print(f"  Execution Accuracy: {ex_score:.0%} - {ex_detail}")
        print(f"  SQL Validity: {val_score:.0%} - {val_detail}")
        print(f"  Security: {sec_score:.0%} - {sec_detail}")

        tier1_score = (ex_score + val_score + sec_score) / 3

        return tier1_score, {
            "execution_accuracy": (ex_score, ex_detail),
            "validity": (val_score, val_detail),
            "security": (sec_score, sec_detail),
        }

    def _evaluate_tier2(self, vector_calls: List[Dict], test_case: Dict):
        """Evaluate Tier 2: Retrieval Precision."""
        print("\n--- Tier 2: Retrieval Precision ---")

        vec_score, vec_detail = self.retrieval.evaluate_vector_search_usage(
            vector_calls, test_case
        )
        rel_score, rel_detail = self.retrieval.evaluate_retrieval_relevance(
            vector_calls, test_case
        )

        print(f"  Vector Search Usage: {vec_score:.0%} - {vec_detail}")
        print(f"  Retrieval Relevance: {rel_score:.0%} - {rel_detail}")

        # Tier 2 score based on vector search usage and relevance only
        tier2_score = (vec_score + rel_score) / 2

        return tier2_score, {
            "vector_search_usage": (vec_score, vec_detail),
            "retrieval_relevance": (rel_score, rel_detail),
        }

    def _evaluate_tier3(
        self,
        response: str,
        response_amounts: List[float],
        transaction_ids: List[int],
        test_case: Dict,
    ):
        """Evaluate Tier 3: Response Quality."""
        print("\n--- Tier 3: Response Quality ---")

        faith_score, faith_detail = self.response.evaluate_faithfulness(
            response, test_case["golden_output"], transaction_ids
        )
        amount_score, amount_detail = self.response.evaluate_amount_accuracy(
            response_amounts, test_case, test_case["golden_output"]
        )

        print(f"  Faithfulness: {faith_score:.0%} - {faith_detail}")
        print(f"  Amount Accuracy: {amount_score:.0%} - {amount_detail}")

        # Weight: faithfulness/amounts (100%)
        quality_score = (faith_score + amount_score) / 2
        tier3_score = quality_score

        return tier3_score, {
            "faithfulness": (faith_score, faith_detail),
            "amounts": (amount_score, amount_detail),
        }

    def _print_summary(self, tier1: float, tier2: float, tier3: float, overall: float):
        """Print evaluation summary."""
        print(f"\n--- Overall Score: {overall:.1%} ---")
        print(f"  Tier 1 (Functional): {tier1:.1%}")
        print(f"  Tier 2 (Retrieval): {tier2:.1%}")
        print(f"  Tier 3 (Response): {tier3:.1%}")

    def generate_report(
        self, results: List[EvaluationResult], model_name: str = None
    ) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        header = "EVALUATION REPORT"
        if model_name:
            header += f" - Model: {model_name}"
        report.append(header)
        report.append("=" * len(header))
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        avg_overall = sum(r.overall_score for r in results) / total_tests
        avg_tier1 = sum(r.tier1_score for r in results) / total_tests
        avg_tier2 = sum(r.tier2_score for r in results) / total_tests
        avg_tier3 = sum(r.tier3_score for r in results) / total_tests
        avg_latency = sum(r.latency_seconds for r in results) / total_tests

        report.append("\nSummary:")
        report.append(
            f"  Tests Passed: {passed_tests}/{total_tests} ({passed_tests / total_tests:.1%})"
        )
        report.append(f"  Average Overall Score: {avg_overall:.1%}")
        report.append(f"  Average Latency: {avg_latency:.2f}s")
        report.append(f"  Average Tier 1 (Functional): {avg_tier1:.1%}")
        report.append(f"  Average Tier 2 (Retrieval): {avg_tier2:.1%}")
        report.append(f"  Average Tier 3 (Response): {avg_tier3:.1%}")

        # Individual test results

        report.append("Individual Test Results:")

        for result in results:
            status = "PASS" if result.passed else "FAIL"
            report.append(f"\n{result.test_id}: {result.question}")
            report.append(
                f"  Status: {status} (Latency: {result.latency_seconds:.2f}s)"
            )
            report.append(
                f"  Overall: {result.overall_score:.1%} | "
                f"T1: {result.tier1_score:.1%} | "
                f"T2: {result.tier2_score:.1%} | "
                f"T3: {result.tier3_score:.1%}"
            )

            # Show failures
            if not result.passed:
                report.append("  Issues:")
                for tier_name, tier_data in result.details.items():
                    if tier_name in ["tier1", "tier2", "tier3"]:
                        for metric, values in tier_data.items():
                            # Handle both 2-tuple (score, detail) and 4-tuple formats
                            if len(values) >= 2:
                                score, detail = values[0], values[1]
                                if score is not None and score < 1.0:
                                    report.append(f"    - {metric}: {detail}")

        return "\n".join(report)

    def close(self):
        """Clean up resources."""
        self.functional.close()

import json
import argparse
from src.agent import RagSqlAgent
from evaluation import AgentEvaluator


def main():
    """Run evaluation on all test cases."""

    parser = argparse.ArgumentParser(description="SQL-RAG Agent Evaluation Framework")
    parser.add_argument(
        "--model", type=str, default="qwen3:4b", help="Model name to use for evaluation"
    )
    parser.add_argument(
        "--reasoning", action="store_true", help="Enable reasoning mode"
    )
    args = parser.parse_args()

    print(f"SQL-RAG Agent Evaluation Framework (Model: {args.model})")

    # Load test cases
    print("\nLoading ground truth test cases...")
    with open("evaluation/ground_truth_test_cases.json", "r") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases")

    # Initialize evaluator
    evaluator = AgentEvaluator(db_path="transactions.db")

    # Run evaluations
    results = []
    current_client_id = None
    agent = None

    for test_case in test_cases:
        # Reinitialize agent if client_id changes
        if test_case["client_id"] != current_client_id:
            current_client_id = test_case["client_id"]
            print(
                f"\nInitializing agent for client {current_client_id} using model {args.model}..."
            )
            agent = RagSqlAgent(
                client_id=current_client_id,
                model_name=args.model,
                reasoning=args.reasoning,
            )
        elif hasattr(agent, "reset_conversation"):
            # Prevent cross-test contamination for same client_id
            agent.reset_conversation()

        result = evaluator.evaluate_test_case(agent, test_case)
        results.append(result)

    # Generate and save report
    report = evaluator.generate_report(results, model_name=args.model)
    print(report)

    with open("evaluation/evaluation_report.txt", "w") as f:
        f.write(report)

    print("Report saved to evaluation/evaluation_report.txt")

    # Save detailed results as JSON
    results_dict = [
        {
            "test_id": r.test_id,
            "question": r.question,
            "passed": r.passed,
            "overall_score": r.overall_score,
            "latency_seconds": r.latency_seconds,
            "tier1_score": r.tier1_score,
            "tier2_score": r.tier2_score,
            "tier3_score": r.tier3_score,
            "generated_sql": r.details.get("generated_sql", ""),
            "response": r.details.get("response", "")[:500],
        }
        for r in results
    ]

    # Include metadata in JSON results
    results_output = {
        "metadata": {"model": args.model, "reasoning": args.reasoning},
        "results": results_dict,
    }

    with open("evaluation/evaluation_results.json", "w") as f:
        json.dump(results_output, f, indent=2)

    print("Detailed results saved to evaluation/evaluation_results.json")

    # Cleanup
    evaluator.close()


if __name__ == "__main__":
    main()

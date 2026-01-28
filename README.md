# Financial Transaction Assistant

[![CI](https://github.com/afifhaziq/TransactionsLangchain/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/afifhaziq/TransactionsLangchain/actions/workflows/ci.yml)
[![CodeQL](https://github.com/afifhaziq/TransactionsLangchain/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/afifhaziq/TransactionsLangchain/actions/workflows/codeql.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A hybrid RAG-SQL financial assistant that answers natural language questions about bank transactions with 90%+ accuracy.

## Features

- **Hybrid RAG-SQL Agent**: Combines semantic search with SQL for accurate transaction queries
- **Enterprise-Grade Security**: Tool-level enforcement prevents SQL injection and cross-client access
- **Comprehensive Evaluation**: 3-tier evaluation framework (Functional, Retrieval, Response Quality)
- **Local-First**: Runs entirely on local hardware using Ollama models
- **High Accuracy**: 90%+ pass rate with proper prompt engineering
- **Well-Tested**: Automated CI/CD with security compliance tests
- **Production-Ready**: Includes both CLI and Streamlit web interfaces

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/afifhaziq/TransactionsLangchain.git
cd TransactionsLangchain

# 2. Install dependencies
uv sync  # or pip install -r requirements.txt

# 3. Install and setup Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b

# 4. Prepare your data (CSV format with columns: clnt_id, txn_id, txn_date, desc, merchant, cat, amt, acc_id, bank_id)
# Place your data.csv in the root directory

# 5. Ingest data
uv run src/ingest_sql.py       # Create SQLite database
uv run src/ingest_vector.py    # Create vector embeddings

# 6. Run the assistant
uv run main.py                 # CLI interface
# OR
uv run streamlit run streamlit_app.py  # Web interface
```

## Requirements

### System Requirements
- Python 3.12+
- 8GB+ RAM (for running Ollama models)
- 10GB+ disk space (for models and vector store)
- Linux/macOS/Windows (WSL2)

### Python Dependencies
- LangChain 0.3+
- ChromaDB
- SQLAlchemy
- Streamlit
- Ollama Python client
- pytest (for testing)

See [pyproject.toml](pyproject.toml) for complete dependency list.

### Data Format
Your CSV file should contain the following columns:
- `clnt_id`: Client ID (integer)
- `txn_id`: Transaction ID (string)
- `txn_date`: Transaction date (format: DD/MM/YYYY)
- `desc`: Transaction description (string)
- `merchant`: Merchant name (string)
- `cat`: Category (string)
- `amt`: Amount (float, negative for spending, positive for income)
- `acc_id`: Account ID (string)
- `bank_id`: Bank ID (string)

## Problem Statement

This project implements a financial assistant to help users make informed financial decisions. The assistant can answer questions about bank transactions in natural language (English). Example questions include:

1. How much did I spend last week?
2. What is the amount I have spent on Uber in the last 5 months?

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution: Hybrid RAG-SQL Agent](#1-solution-hybrid-rag-sql-agent)
- [Evaluation Framework](#2-evaluation-framework)
- [Running the Demo](#3-running-the-demo)
- [Experimental Setup](#4-experimental-setup)
- [Prompt Engineering](#5-prompt-engineering-highlights)
- [Evaluation Results](#evaluation-results)
- [Challenges & Solutions](#6-challenges--solutions)
- [CI/CD & Testing](#cicd--testing)
- [Conclusion](#conclusion)

---

## Initial Analysis

As the first step of any AI-based project, the focus is on understanding the dataset, as the solution will always be dependent on the data structure, columns, and relationships. An analysis was performed using **ydata_profiling** to understand the dataset better.

```python
import ydata_profiling
import pandas as pd
df = pd.read_csv("data.csv")
profile = ydata_profiling.ProfileReport(df)
profile.to_file("analysis.html")
```

This tool helps to understand the dataset better and visualize the data. Key findings include that each client has a unique **clnt_id**, each transaction has a unique **txn_id** with respect to the client, and **bank_id** and **acc_id** are unique with regards to the client.

## Initial Thought

The initial plan was to create a RAG agent that can answer questions about the dataset. However, several considerations emerged:

1. The dataset is mostly structured. Some columns like **desc**, **cat** and **merchant** are unstructured which could leverage the vector search from RAG.
2. Vector search is unreliable since it uses semantic search to look for the most similar values in the dataset using embeddings. The results are not always accurate and could be misleading. In a business setting, this could lead to wrong decisions and could be costly.
3. The dataset is in the format of a .csv file which could be converted to a database and leverage SQL queries to answer questions.

Research on the SQL agent approach revealed several approaches:
1. The first approach is to use the SQL agent where the agent uses the dataset to answer questions. However, this approach is not ideal since the model struggles to generate the correct SQL query with questions like "How much did I spend on the convenience store?". The model could generate a query like "SELECT SUM(amt) FROM transactions WHERE cat = 'convenience store'" which is not correct since the model misses the semantic information of the question.
2. The second approach is to use the RAG agent where the agent uses vector search to answer questions. However, it is not reliable since the results will be based on the top n results from the semantic search. Questions like "Calculate my total spending on the convenience store?" will not be able to retrieve all of the related transactions given the number of transactions exceed the number of n from vector search.
3. The third approach is to use the hybrid approach where the agent uses vector search to perform semantic search for columns like **desc**, **cat** and **merchant**. Based on the output of this tool, the model can generate an SQL query to retrieve the related transactions. This approach is the most reliable and accurate since the vector search helps in finding the correct categories and merchants semantically.

---

## 1. Solution: Hybrid RAG-SQL Agent
## Files Structure

The files structure of the project is as follows:

```
TransactionsLangchain/
├── src/
│   ├── agent.py              # Main RAG-SQL agent
│   ├── ingest_sql.py         # CSV → SQLite ingestion
│   └── ingest_vector.py      # ChromaDB vector store creation
├── evaluation/
│   ├── __init__.py           # Package initialization
│   ├── tier1_functional.py   # SQL execution & security
│   ├── tier2_retrieval.py    # RAG quality + precision/recall
│   ├── tier3_response.py     # Response faithfulness + accuracy
│   ├── evaluator.py          # Main orchestrator
│   ├── extractors.py         # Output parsing
│   ├── ground_truth_test_cases.json  # 10 test cases
│   ├── results/  # Sample results of the evaluation
│   └── README.md         # Evaluation README
├── main.py                   # CLI interface
├── streamlit_app.py          # Web UI
├── evaluate_agent.py         # Run evaluation
├── transactions.db           # SQLite database
├── chroma_db/                # Vector store
├── README.md                 # This file
├── pyproject.toml            # Project configuration
└── uv.lock                   # Dependency lock file

```


### Why Hybrid?

- **RAG**: Semantic search for description/category/merchant names
- **Agentic**: LLM decides when to use vector search vs direct SQL vs direct response
- **Reliability**: The agents will be able to retrieve real transactions based on the database

### Architecture Flow

```
User Query: "Show me grocery purchases in August"
    ↓
Agent Decision: "Query mentions 'grocery' → Need vector search"
    ↓
Tool 1: vector_search("grocery")
    Returns: "category: 'Supermarkets and Groceries'"
    ↓
Agent Decision: "Use returned category in SQL WHERE clause"
    ↓
Tool 2: sql_db_query("""
    SELECT * FROM transactions 
    WHERE clnt_id = <user_client_id> 
      AND cat = 'Supermarkets and Groceries'
      AND txn_date LIKE '%/08/2023%'
      AND amt < 0
    LIMIT 5
    """)
    Returns: [(txn_id, date, desc, merchant, cat, amt, acc_id, bank_id), ...]
    ↓
Response: "You had X grocery purchases in August totaling $Y"
```

### Key Design Decisions

#### 1. Vector Search: 15 Results + Category-First Ranking

Testing revealed that vector search with n_results=5, returning only top 5 matches, caused categories to appear later in semantic ranking, missing critical matches. For instance:

**Search term: "grocery"**

   - 1. **"GROCERY" (merchant)**           - distance: 0.0821  # Exact substring
   - 2. **"Progress Grocery" (merchant)** - distance: 0.1338  # Contains "Grocery"
   - 8. **"Supermarkets and Groceries" (category)** - distance: 0.1632 # Semantic match
   
With 5 results, categories might be missed by the model since the results are based on the top n results from the semantic search. This could be due to the small embedding model used (qwen3-embedding:0.6b) and the limited results. Therefore, 15 results are used for the vector search and the results are reranked with categories appearing first.

**Why category first ranking?**
   - User intent analysis: "grocery" almost always means category, not specific merchant based on testing and data observation.
   - Reranking ensures the final output prioritizes categories over merchants.
   - Merchant-specific queries ("Walmart") still work correctly.
   - Since merchants are subcategories of categories, this decision does not significantly affect the accuracy of results.

#### 2. Why Only 3 Columns in Vector Store?

Only **desc**, **merchant**, and **cat** are embedded because:
- These columns contain semantic information requiring fuzzy matching
- Numeric/date columns (amt, txn_date) are better handled by SQL directly
- Reduces vector store size (105K samples vs 257K transactions samples) - 59% reduction in size
- Improves search speed (~70ms per query)

#### 3. Security: Tool-Level Enforcement

```python
@tool("sql_db_query")
def sql_db_query(query: str, runtime: ToolRuntime[Context]) -> str:
    ctx_client_id = runtime.context.clnt_id
    
    # Security check: Enforce client isolation
    if str(ctx_client_id) not in query:
        return f"Error: Query must filter by clnt_id = {ctx_client_id}"
    
    return db.run(query)
```

**Why Tool-Level?**

One of the first challenges was access control for user accounts. Since the database is shared across users, there is a significant security concern for cross-client data access. Testing revealed that even when the system prompt specifies the client_id, the model can bypass it using common prompt injection techniques. Therefore, client_id enforcement is implemented at the tool level. The implementation verifies the clnt_id and injects it into the SQL query tools to ensure the model always filters by the client_id with WHERE clnt_id = {client_id}. This implementation is more secure and reliable. 

---

## 2. Evaluation Framework

### Why Evaluation Matters

The evaluation framework is crucial for the success of the financial assistant. Issues that could arise include:
- Wrong amounts can cause users to make bad financial decisions
- Hallucinated transactions can cause loss of trust
- Security breaches can cause regulatory violations
- Inconsistent formatting can cause confusion

A **three-tier evaluation system** was created that compares the output of tools, model's output, and the golden output. The evaluation code is in the `/evaluation` folder with ground truth in `/evaluation/ground_truth_test_cases.json`.

### Three-Tier Evaluation System

#### Tier 1: Functional Correctness (50% weight)

The weightage of this tier is 50% since financial accuracy is non-negotiable. Wrong SQL queries will lead to wrong answers and can cause significant issues. Although there are multiple SQL queries that can be functionally equivalent, execution accuracy is the most important metric. The evaluation code is in the **/evaluation/tier1_functional.py** file.

**Metrics** (from `/evaluation/tier1_functional.py`):

1. **Execution Accuracy**: Does generated SQL return exact same results as golden SQL?
   - Compares actual database results, not SQL strings
   - Multiple SQL queries can be functionally equivalent

2. **SQL Validity**: Does SQL execute without syntax errors?

3. **Security Compliance**: Does SQL include `clnt_id` filter?
   - Also checks for dangerous operations (DROP, DELETE, UPDATE)
   - Detects SQL injection patterns (UNION SELECT, OR 1=1, etc.)

**Example Test Case**:
```json
{
  "test_id": "TC001",
  "question": "How much did I spend in August 2023?",
  "golden_sql": "SELECT SUM(amt) FROM transactions WHERE clnt_id = <client_id> AND amt < 0 AND txn_date LIKE '%/08/2023%'",
  "golden_output": [{"total_spending": -638532.93}],
  "expected_amount": -638532.93
}
```

#### Tier 2: Retrieval Precision (25% weight)

A weightage of 25% is assigned to this tier since RAG quality affects efficiency and accuracy, but is secondary to SQL correctness as not all queries are related to description, categories, or merchants. The evaluation code is in the **/evaluation/tier2_retrieval.py** file.

**Metrics** (from `/evaluation/tier2_retrieval.py`):

1. **Vector Search Usage**: Is vector search called when needed?
   - Compares against `need_vector` flag in test case
   - Score: 1.0 if usage matches expectation, 0.0 if missing when needed
   - Score: 0.5 if called but not necessary

2. **Retrieval Relevance**: Does search query contain relevant terms?
   - Checks if search query contains `expected_search_terms` from test case
   - Example: User asks "grocery" → Search should contain "grocery" or similar
   - Validates that the agent searches for semantically correct terms

**Tier 2 Scoring Formula**:
```python
# From evaluator.py lines 191-192
tier2_score = (vec_score + rel_score) / 2
```

**Why This Matters**: Ensures the agent uses vector search appropriately and searches for the right terms, which directly impacts SQL query accuracy.

#### Tier 3: Response Quality (25% weight)

A weightage of 25% is assigned to this tier since response quality is important for user experience. In some cases, the model will try to alter amounts and transaction IDs during formatting. This metric ensures the model is not altering transaction details. The evaluation code is in the **/evaluation/tier3_response.py** file.

**Metrics** (from `/evaluation/tier3_response.py`):

1. **Faithfulness**: No hallucinations or placeholders
   - Checks for placeholder text: "$X.XX", "[actual date]", "[transaction]"
   - Detects fabricated sequential IDs (1, 2, 3, 4, 5)
   - Verifies transaction IDs mentioned in response exist in golden output
   - Prevents model from fabricating transactions

2. **Amount Accuracy**: Numbers match expected values
   - Tolerance: ±$0.01 (rounding acceptable)
   - Checks `expected_amount`, `expected_spending`, `expected_income`
   - Ensures financial figures are accurate to the cent

**Tier 3 Scoring Formula**:
```python
# From evaluator.py lines 219-221
quality_score = (faith_score + amount_score) / 2
tier3_score = quality_score
```


### Overall Scoring

```python
# From evaluator.py lines 80-86
overall_score = (
    tier1_score * 0.5 +   
    tier2_score * 0.25 +  
    tier3_score * 0.25    
)

passed = overall_score >= 0.8  # 80% threshold to pass
```

**Scoring Breakdown**:
- **Tier 1 (50%)**: Functional correctness
- **Tier 2 (25%)**: Retrieval quality
- **Tier 3 (25%)**: Response quality

**Pass Criteria**: Overall score ≥ 80% ensures high-quality responses across all dimensions.

### Ground Truth Generation

Ground truth is important - if the method of obtaining it is wrong, the evaluation falls apart. Therefore, actual database queries are used to generate the ground truth. This ensures the evaluation is accurate and not biased by synthetic data. Sample test cases are included in the **/evaluation/ground_truth_test_cases.json** file.

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///transactions.db')

# Test Case 1: Simple aggregation
tc001_sql = """
SELECT SUM(amt) as total_spending 
FROM transactions 
WHERE clnt_id = <client_id> 
  AND amt < 0 
  AND txn_date LIKE '%/08/2023%'
"""
tc001_result = pd.read_sql(tc001_sql, engine)

test_cases.append({
    "test_id": "TC001",
    "question": "How much did I spend in August 2023?",
    "golden_sql": tc001_sql,
    "golden_output": tc001_result.to_dict('records'),
    "expected_amount": abs(tc001_result['total_spending'][0])
})
```

**Why Real Ground Truth?**
- Tests against actual database behavior
- Catches edge cases (NULL values, date formats)
- No synthetic data bias
- Validates production-ready performance

### Test Coverage (10 Test Cases)

All test cases use a sample client with 189,986 transactions and $2.47M spending:

| ID | Question | Type | Expected Result |
|----|----------|------|-----------------|
| TC001 | "How much did I spend in August 2023?" | Simple aggregation | $638,532.93 |
| TC002 | "How much did I spend on restaurants in June 2023?" | Category filter | $32,394.36 |
| TC003 | "Show me my first 3 restaurant transactions" | Transaction list | 3 specific transactions |
| TC004 | "Show me top 5 most expensive expenses for groceries" | Amount filter | 5 large purchases |
| TC005 | "What's my total income and spending in July 2023?" | Income vs spending | Spending: $602,027.27<br>Income: $776.68 |
| TC006 | "Show me my Ferrari purchases" | Empty result | 0 transactions |
| TC007 | "Show me my largest Walmart purchases" | Merchant filter | 5 Walmart purchases |
| TC008 | "What's my total spending across all transactions?" | Total aggregation | $2,472,806.78 |
| TC009 | "Show me my ATM withdrawals in June 2023" | Category + Date | 5 ATM withdrawals |
| TC010 | "Compare my spending on restaurants vs groceries" | Multi-category | Groceries: $150,189.55<br>Restaurants: $138,827.69 |

---

## 3. Running the Demo

### Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull qwen3:4b                
ollama pull qwen3-embedding:0.6b
```

### Setup & Run
1. Install dependencies
```bash
# Install uv if not installed
uv sync # Make sure uv is installed

# Alternatively, you can use pip to install the dependencies
pip install -r requirements.txt
```
2. Ingest data
```bash
uv run src/ingest_sql.py      # CSV → SQLite
uv run src/ingest_vector.py   # Embed categories/merchants → ChromaDB
```
3. Run ollama server. If you wish to run the evaluation, you can run this command and skip to the **uv run evaluate_agent.py** step.
```
ollama serve
```
Main CLI interface. This is the headless setup for the assistant.
```python
uv run main.py
```

Alternatively, you can use the following command to run the streamlit app:
```bash
# Alternatively, you can use the following command to run the streamlit app:
# Run Streamlit UI (Optional)
uv run streamlit run streamlit_app.py
# Open http://localhost:8501
```
Run evaluation. Note that this line runs the evaluation on all 10 test cases based on the ground truth in the **/evaluation/ground_truth_test_cases.json** file. 
```bash
# Run with default model (qwen3:4b)
uv run evaluate_agent.py

# Run with different model
uv run evaluate_agent.py --model qwen3:8b --reasoning 

# Output files:
# - evaluation/evaluation_report.txt (human-readable report)
# - evaluation/evaluation_results.json (detailed JSON results)
```
Optional: Run ruff check to ensure the code is formatted correctly and implement best practices.
```bash
# Run ruff check
uv run ruff check
# fix ruff check issues
uv run ruff check --fix
```

### Example Interaction

```
User > How much did I spend on restaurants in June 2023?

================================== Ai Message ==================================
Tool Calls:
  vector_search (call_1)
  Args: query: restaurants

================================= Tool Message =================================
Name: vector_search
- category: 'Restaurants'
- category: 'Fast Food Restaurants'

================================== Ai Message ==================================
Tool Calls:
  sql_db_query (call_2)
  Args: query: SELECT SUM(amt) FROM transactions WHERE clnt_id = <user_client_id> 
                AND cat = 'Restaurants' AND txn_date LIKE '%/06/2023%' AND amt < 0

================================= Tool Message =================================
Name: sql_db_query
[(-32394.36,)]

================================== Ai Message ==================================
You spent $32,394.36 on restaurants in June 2023.
```

---

## 4. Experimental Setup

The system was designed to run on local hardware with moderate specifications. This demonstrates that a performant system can be built with the right implementation and optimization, even on resource-constrained devices.

| Hardware/Software | Configuration |
|----------|---------------|
| CPU | Intel i5-13500HX |
| RAM | 24GB |
| GPU | Nvidia RTX 4050 |
| OS | Linux (Ubuntu 22.04) |
| Backend | Ollama |
| Agentic Framework | LangChain |
| Frontend | Streamlit |
| Vector Store | ChromaDB |

### Model Choice

The experiment leverages Ollama models that are open source and free to use. Model choices are based on performance for general tasks and tool-calling capabilities according to available benchmarks. Smaller models were chosen to minimize latency.

|Model        | Provider                                      |
|-----------------------|------------------------------------------------|
| qwen3:8b    | Ollama |
| qwen3:4b  | Ollama |
| qwen3:1.7b      | Ollama |
| qwen3:0.6b       | Ollama |
| functiongemma:270m              | Ollama |
| qwen3-embedding:8b              | Ollama |
| qwen3-embedding:4b              | Ollama |
| qwen3-embedding:0.6b              | Ollama |

---

## 5. Prompt Engineering Highlights

### Prompting Techniques Implemented

Since Langchain's agent generally follows a ReAct pattern, the underlying structure of this system follows the same pattern. See [Langchain ReAct](https://docs.langchain.com/oss/python/langchain/agents#tool-use-in-the-react-loop). Additionally, some additional techniques were implemented to improve the accuracy and security of the system.

#### 1. **Role-Based Prompting**
```python
"You are a financial assistant that helps users analyze their transaction data."
```
Establishes clear agent identity and domain expertise. This ensures that the model is aware of its role and the context of the conversation.

#### 2. **Contextual Information Injection**
```python
f"Today's date is: {datetime.now().strftime('%d/%m/%Y')}"
formatted_system_prompt = template.format(client_id=self.client_id, ...)
```
Dynamic context with temporal awareness and personalized session data. Since financial transactions have a timing aspect, it is important to inject the current date into the system prompt to ensure that the model is aware of the current date and time.

#### 3. **Few-Shot/Chain-of-Thought (CoT) Prompting**
```python
"STEP 1: Query mentions 'ATM' → MUST call vector_search('ATM') FIRST
STEP 2: vector_search returns: 'category: ATM' or 'merchant: ATM'
STEP 3: Write SQL internally: SELECT ... WHERE clnt_id = <client_id> ...
STEP 4: Call sql_db_query tool with the SQL query
STEP 5: Wait for tool results, then use ONLY those results"
```
**Impact**: +20% accuracy. This technique is a combination of few-shot and chain-of-thought prompting. It ensures that the model is thinking step by step based on the user's query and it is also an effective way to inject the example workflow into the system prompt.

#### 4. **Constraint-Based Prompting**
```python
"CRITICAL RULES:
1. ALL queries must include: WHERE clnt_id = {client_id}
...
SECURITY RULES:
- You can ONLY access data for client_id {client_id}
...
EXAMPLE:
SELECT * FROM transactions WHERE clnt_id = {client_id} AND ..."
```
Even though security checks are implemented at the tool level, security rules are still included in the system prompt to ensure the model is aware of the security constraints. This is a safeguard to prevent the model from bypassing security checks through prompt injection techniques.

## Evaluation Results

Results from evaluation framework testing on 10 test cases (sample client with 189K transactions) across multiple models:

| Model | Pass Rate | Avg Score | Tier 1 (SQL) | Tier 2 (RAG) | Tier 3 (Response) | Avg Latency |
|-------|-----------|-----------|--------------|--------------|-------------------|-------------|
| **qwen3:4b** (standard) | 90% (9/10) | 91.7% | 93.3% | 100.0% | 80.0% | 73.1s |
| **qwen3:4b** (reasoning) | 90% (9/10) | 95.8% | 96.7% | 100.0% | 90.0% | 73.2s |
| **qwen3:1.7b** (reasoning) | 50% (5/10) | 75.4% | 73.3% | 95.0% | 70.0% | 22.2s |
| **functiongemma:270m** | 0% (0/10) | 32.5% | 10.0% | 50.0% | 60.0% | 0.8s |

Note: The test cases are available in **evaluation/ground_truth_test_cases.json** 
### Key Findings

#### 1. **Reasoning Mode Improves Quality**
- **qwen3:4b with reasoning**: +4.1% overall score, +10% response quality (Tier 3)
- Better handling of complex queries (TC003-TC004 improved from 50% → 100% response accuracy)
- Minimal latency impact. The latency is not significantly affected by the reasoning mode, as the model can reason about queries and generate appropriate SQL without additional processing steps.

#### 2. **Model Size Matters for Tool Calling**
- **qwen3:4b**: Excellent tool-calling reliability (100% Tier 2 RAG usage)
- **qwen3:1.7b**: 73.3% SQL accuracy, struggles with complex multi-step queries. Testing revealed poor performance with longer system prompts. The model often overthinks and forgets instructions in the system prompt.
- **functiongemma:270m**: Failed to generate SQL in 9/10 cases. This is the latest variant from gemma specialized for edge deployment. Despite test failures, the latency is extremely low at 0.8s compared to qwen3:4b at 73.1s. This model is designed to be fine-tuned for specific tasks and shows promising potential for edge deployment.

#### 3. **Common Failure Pattern: Sorting Edge Case**
- **TC009** (ATM withdrawals sorted by smallest amount) failed on both qwen3:4b variants. This test case demonstrates the model's difficulty with edge cases.
- All models sorted by absolute value descending instead of ascending. Models typically get confused with negative values and sort them in the wrong order. This may be because the database stores negative values as spending and the model tries to reformat for final output.
- **Solution**: Try larger models with better reasoning capabilities.


#### 4. **Tier Performance Analysis**
- **Tier 1 (Functional)**: 93.3% - SQL generation highly reliable with proper prompts. In some cases, the model outputs the SQL query to the user instead of the database results, especially common with smaller models.
- **Tier 2 (Retrieval)**: 100% - Perfect RAG tool usage (vector search when needed). CoT and Few-shot prompting significantly improves retrieval accuracy.
- **Tier 3 (Response)**: 80% - Occasional formatting issues, but no fabricated data
- **Security**: 100% compliance - Tool-level enforcement prevents all bypass attempts.

---

## 6. Challenges & Solutions

### Challenge 1: SQL Query Checker Loop

**Problem**: LangChain's `sql_db_query_checker` tool created validation loops with false positives. This is a common issue with smaller models as they often overthink and forget instructions in the system prompt.

**Solution**: Remove checker tool, rely on database error messages.

### Challenge 2: Hallucinated Transaction IDs

**Problem**: Agent fabricated transaction IDs (1, 2, 3, 4, 5) instead of using actual IDs. This is fairly common when the model tries to answer without executing the sql_db_query tool. The model also does this when reformatting details before final response.

**Solution**: Add explicit instruction: "Use ONLY data from sql_db_query tool results" and negative instructions ("do NOT fabricate") to ensure the model is not fabricating data.

---

## CI/CD & Testing

### GitHub Actions Workflows

This repository includes automated CI/CD pipelines to ensure code quality and security:

#### 1. **CI Pipeline** (`.github/workflows/ci.yml`)

The CI pipeline runs on every push and pull request to the main branch, consisting of three jobs:

**Lint Job:**
- Runs Ruff linter to check code style and formatting
- Ensures code follows Python best practices
- Validates code formatting consistency

```bash
# Run locally
uv run ruff check .
uv run ruff format --check .
```

**Test Job:**
- Runs pytest with coverage reporting
- Tests evaluation framework components
- Uploads coverage reports to Codecov
- Only runs if linting passes

```bash
# Run locally
uv run pytest tests/ -v --cov=evaluation --cov=src
```

**Security Job:**
- Matrix testing for SQL injection prevention
- Tests 9 different attack vectors:
  - Valid queries (should pass)
  - SQL injection patterns (OR 1=1, UNION SELECT)
  - Dangerous operations (DROP, DELETE, UPDATE)
  - Missing client_id filters
- Ensures tool-level security enforcement works correctly

#### 2. **CodeQL Security Scanning** (`.github/workflows/codeql.yml`)

- Automated security vulnerability scanning
- Runs on push, pull requests, and weekly schedule
- Uses GitHub's security-extended query suite
- Scans Python code for common security issues

### Running Tests Locally

```bash
# Install dependencies
uv sync --dev

# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=evaluation --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_tier1_functional.py -v

# Run security compliance tests
uv run pytest tests/test_security.py -v
```

### Test Structure

```
tests/
├── test_tier1_functional.py   # SQL execution & security tests
├── test_tier2_retrieval.py    # RAG quality tests
├── test_tier3_response.py     # Response quality tests
└── test_security.py           # Security compliance tests
```

---

## Conclusion

This project demonstrates a financial assistant capable of handling user queries for transaction data.

### Lessons Learned

1. **Hybrid approaches**: RAG + SQL ensure a reliable and accurate system.
2. **Prompt engineering is critical**: Explicit workflows work best when paired with context information.
3. **Evaluation must be rigorous**: Three-tier system catches issues single metrics miss
4. **Security by design**: Tool-level enforcement and prompt-level enforcement are both important for system security.
5. **Small models + good prompts > large models + bad prompts**

## What Didn't Work

1. **VLLM Integration**: Due to resource constraints, VLLM could not be used for inference.
2. **Pydantic AI**: Attempted use of Pydantic AI for the agentic framework, but LangChain's framework provided more useful tools. Pydantic AI does offer better scalability and reliability for production systems.
3. **Nemo Guardrails**: Attempted use caused latency issues and in some cases blocked valid responses.

### Future Improvements

1. **Inference Performance**: Ollama is a good starting point for local deployment. However, it is not as efficient as serving engines such as VLLM. Possible improvements include integration with VLLM and LMCache for better performance.
2. **Conversation Memory**: Add chat history for multi-turn queries. Use supermemory for lightweight and efficient conversation memory.
3. **LLM-as-a-Judge**: Use LLM-as-a-Judge to evaluate the agent's performance.
4. **Larger Models**: Upgrade to closed-source models with better tool-calling capabilities via API calls.
5. **LangSmith/wandb/logfire Integration**: Track evaluation runs for continuous improvement

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Install dependencies (`uv sync --dev`)
4. Make your changes and add tests
5. Run tests (`uv run pytest`)
6. Run linting (`uv run ruff check . && uv run ruff format .`)
7. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
8. Push to the branch (`git push origin feature/AmazingFeature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the agentic framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Qwen Team](https://github.com/QwenLM/Qwen) for the open-source models

---

**Note**: This repository does not include transaction data or databases. You must provide your own dataset in CSV format with the required schema (see evaluation/ground_truth_test_cases.json for expected structure).


import chromadb
from chromadb.utils import embedding_functions
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime
from datetime import datetime
from dataclasses import dataclass


class Context(BaseModel):
    clnt_id: int | None = Field(description="The client ID for the user session")


@dataclass
class TransactionInfo:
    clnt_id: int | None
    bank_id: int
    acc_id: int
    txn_id: int
    txn_date: str
    desc: str
    amt: str  # Display with $ hence string
    cat: str
    merchant: str


@dataclass
class Response:
    summary: str
    details: list[TransactionInfo] | None


class RagSqlAgent:
    def __init__(
        self,
        client_id,
        db_path="transactions.db",
        chroma_path="./chroma_db",
        model_name="qwen3:4b",
        reasoning=False,
    ):
        self.client_id = client_id
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.top_k = 5
        self.reasoning = reasoning
        # Initialize components
        self.db = SQLDatabase.from_uri(
            f"sqlite:///{self.db_path}", sample_rows_in_table_info=0
        )
        self.llm = ChatOllama(
            model=self.model_name, temperature=0.5, reasoning=self.reasoning
        )
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.sql_tools = self.toolkit.get_tools()
        self.setup_vector_store()
        self.setup_agent()
        self.reset_conversation()

    def setup_vector_store(self):
        """Initialize ChromaDB connection"""
        client = chromadb.PersistentClient(path=self.chroma_path)

        # Use same embedding function as ingestion
        ef = embedding_functions.OllamaEmbeddingFunction(
            model_name="qwen3-embedding:0.6b",
            url="http://localhost:11434/api/embeddings",
            timeout=1000,
        )

        self.collection = client.get_collection(
            name="transactions_metadata", embedding_function=ef
        )

    def setup_agent(self):
        """Create the LangChain SQL Agent with custom system prompt"""

        self.table_info = TransactionInfo.__dict__

        db = self.db
        vector_store = self.collection

        @tool(
            "vector_search",
            description='Search for exact database values that match a category, merchant, or product name. Use this BEFORE writing SQL queries when the user mentions any category or merchant name. Pass the search term directly (e.g., "furniture", "coffee"), NOT a SQL query.',
            return_direct=False,
        )
        def vector_search_tool(query: str, n_results: int = 15) -> str:
            """
            MANDATORY FIRST STEP: Search vector store for exact database values matching categories, merchants, or product types.

            WHEN TO USE:
            - User mentions ANY category (e.g., "restaurants", "groceries", "gas", "ATM", "shopping")
            - User mentions ANY merchant/store name (e.g., "Walmart", "Starbucks", "Target")
            - User mentions product types (e.g., "coffee", "food", "furniture")
            - User asks about transaction types (e.g., "ATM withdrawals", "restaurant spending")

            WHAT IT DOES:
            Finds the ACTUAL values stored in the database that match the user's search term.
            Example: User says "grocery" → returns "Supermarkets and Groceries" (the exact category name in DB)
            Example: User says "ATM" → returns "ATM" or "ATM Withdrawal" (the exact category/description in DB)

            RETURNS:
            A formatted string with exact database values to use in SQL WHERE clauses.
            Prioritizes category matches over merchant matches.

            WORKFLOW:
            1. User asks about transactions with category/merchant → CALL THIS TOOL FIRST
            2. Use the returned exact values in your SQL WHERE clause
            3. Then call sql_db_query with the correct category/merchant values
            """
            print(f"Vector search query: {query}")
            try:
                # Search with higher n_results to get more comprehensive matches
                results = vector_store.query(query_texts=[query], n_results=15)

                if not results["documents"]:
                    return ""

                # Separate categories and merchants, prioritize categories
                categories = []
                merchants = []

                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    col = metadata.get("column", "unknown")
                    if col == "cat":
                        categories.append(f"- category: '{doc}'")
                    elif col == "merchant":
                        merchants.append(f"- merchant: '{doc}'")

                # Combine results with categories first
                suggestions = categories + merchants

                if not suggestions:
                    return ""

                return "\n".join(suggestions)

            except Exception as e:
                print(f"Vector search error: {e}")
                return ""

        @tool(
            "sql_db_query",
            description="Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again",
        )
        def sql_db_query(query: str, runtime: ToolRuntime[Context]) -> str:
            """Execute a SQL query with client_id validation from context."""

            # Access client_id from the runtime context
            ctx_client_id = runtime.context.clnt_id

            # Security check: Ensure the query filters by the authenticated client_id
            if str(ctx_client_id) not in query:
                return f"Error: Security violation. Query must filter by clnt_id = {ctx_client_id}"

            try:
                return db.run(query)
            except Exception as e:
                return f"Error: {e}"

        # Replace the default sql_db_query with our secure version and remove unreliable checker
        self.sql_tools = [
            t
            for t in self.sql_tools
            if t.name not in ["sql_db_query", "sql_db_query_checker"]
        ]
        self.sql_tools.append(sql_db_query)

        # Store the vector search tool
        self.vector_search_tool = vector_search_tool

        self.system_prompt_template = (
            "You are a financial assistant that helps users analyze their transaction data.\n"
            f"Today's date is: {datetime.now().strftime('%d/%m/%Y')} (DD/MM/YYYY)\n"
            "You have access to tools to query a transaction database. Use these tools to answer user questions.\n\n"
            "CONVERSATION MEMORY (IMPORTANT):\n"
            "- You are given the prior conversation in the `messages` you receive.\n"
            "- If the user asks meta questions about the conversation (e.g., 'what was my previous question?', 'repeat that', 'summarize our chat'), answer DIRECTLY from the message history.\n"
            "- For these meta conversation questions: DO NOT call any tools and DO NOT claim you lack access to chat history.\n\n"
            "Do not explain what you are going to do. Just use the tools if needed and return the output based on user's queries\n"
            "TOOL USAGE RULES:\n"
            "- When user mentions categories, merchants, stores, or product types (e.g., 'restaurants', 'ATM', 'Walmart', 'coffee'), call vector_search tool FIRST to find exact database values\n"
            "- After getting vector_search results, use those exact values in your SQL query\n"
            "- Always call sql_db_query tool to execute SQL and get actual data - NEVER write SQL in your response text\n"
            "- Wait for tool results before responding to the user\n"
            "- Use ONLY the data returned by tools in your response for TRANSACTION questions\n\n"
            "EXAMPLE WORKFLOW:\n"
            "User: 'Show me my ATM withdrawals in June 2023'\n"
            "STEP 1: Query mentions 'ATM' → MUST call vector_search('ATM') FIRST\n"
            "STEP 2: vector_search returns: 'category: ATM' or 'merchant: ATM'\n"
            f"STEP 3: Write SQL internally: SELECT ... FROM transactions WHERE clnt_id = {self.client_id} AND (cat = 'ATM' OR desc LIKE '%ATM%') AND txn_date LIKE '%/06/2023%' AND amt < 0\n"
            "STEP 4: Call sql_db_query tool with the SQL query (CRITICAL: Never write SQL directly in response)\n"
            "STEP 5: Wait for tool results, then use ONLY those results to answer user\n\n"
            "CRITICAL RULES:\n"
            f"1. You can ONLY access data for client_id {self.client_id}. ALL queries must include: WHERE clnt_id = {self.client_id}\n"
            "2. Dates are stored as DD/MM/YYYY format. For month queries, use: txn_date LIKE '%/MM/YYYY%'\n"
            "   Example: August 2023 → txn_date LIKE '%/08/2023%'\n"
            f"3. Unless the user specifies a specific number of examples, always limit your query to at most {self.top_k} results.\n"
            "4. If user asks about spending/withdrawals, use 'amt < 0' else if user asks about deposits/income, use 'amt > 0'. NEVER use ABS(amt) in your SQL queries.\n"
            "5. If user asks about smallest or biggest spending or income, use the absolute amount to sort in the final response."
            "6. For transaction lists/details, SELECT ALL relevant fields: txn_id, txn_date, desc, merchant, cat, amt, acc_id, bank_id\n"
            "7. MANDATORY: NEVER generate SQL queries directly in your response. ALWAYS use the sql_db_query tool.\n"
            "   FORBIDDEN: Do not write 'SELECT...' statements in your text responses.\n"
            "   REQUIRED: All transaction data must come from sql_db_query tool results.\n"
            "8. Display the final amount in your response with their sign as they appear in database results (negative for spending, positive for income).\n"
            "9. VECTOR SEARCH IS MANDATORY when user mentions:\n"
            "   - Categories: 'ATM', 'restaurants', 'groceries', 'gas', 'shopping', 'travel', 'entertainment'\n"
            "   - Merchants: 'Walmart', 'Starbucks', 'Target', 'McDonald's', any store name\n"
            "   - Products: 'coffee', 'food', 'furniture', 'clothing'\n"
            "   NEVER write SQL queries with guessed category/merchant values. ALWAYS use vector_search first to get exact database values.\n\n"
            "RESPONSE RULES:\n"
            "- Answer transaction questions with actual data from sql_db_query tool results only\n"
            "- NEVER include SQL queries or database syntax in your response text\n"
            "- For transaction lists, show ALL available details in structured format\n"
            "- Use natural language, not robotic phrases\n"
            "- If no data found, say 'I couldn't find any transactions matching your criteria'\n\n"
            "- NEVER provide the tool call in your response. Only provide the final response to the user.\n\n"
            "TRANSACTION FORMATTING (MANDATORY for list/detail queries):\n"
            "When showing transaction details, you MUST include ALL fields in this exact format for EACH transaction:\n"
            "- Transaction ID: [txn_id] (numeric)\n"
            "- Date: [txn_date] (exact format from database)\n"
            "- Description: [desc]\n"
            "- Merchant: [merchant] (or 'N/A' if null)\n"
            "- Category: [cat]\n"
            "- Amount: $[amt] (show with sign as in database. Round to 2 decimal places)\n"
            "- Account ID: [acc_id] (numeric)\n"
            "- Bank ID: [bank_id] (numeric)\n"
            "DO NOT abbreviate or omit any fields. Each transaction must show complete information.\n\n"
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[self.vector_search_tool] + self.sql_tools,
            system_prompt=self.system_prompt_template,
            context_schema=Context,
            response_format=Response,
        )
        self.ctx = Context(clnt_id=self.client_id)

    def reset_conversation(self):
        """Clear chat history for multi-turn conversations."""

        self.messages: list[dict] = []

    def _invoke_sync(self, query: str, remember: bool = True):
        """Helper method for synchronous invocation."""
        messages = [*self.messages, {"role": "user", "content": query}]

        result = self.agent.invoke(
            {"messages": messages},
            context=self.ctx,
            reasoning=self.reasoning,
        )

        if remember:
            assistant_text = ""
            try:
                assistant_text = (
                    getattr(result["messages"][-1], "content", "")
                    if isinstance(result, dict)
                    else ""
                )
            except Exception:
                assistant_text = ""
            self.messages = [
                *self.messages,
                {"role": "user", "content": query},
                {"role": "assistant", "content": str(assistant_text)},
            ]

        return result

    def _stream_generator(self, query: str, remember: bool = True):
        """Generator method for streaming responses."""
        messages = [*self.messages, {"role": "user", "content": query}]

        last_step = None
        for step in self.agent.stream(
            {"messages": messages},
            context=self.ctx,
            stream_mode="values",
            reasoning=self.reasoning,
        ):
            last_step = step
            yield step

        if remember:
            assistant_text = ""
            try:
                if (
                    isinstance(last_step, dict)
                    and "messages" in last_step
                    and last_step["messages"]
                ):
                    assistant_text = getattr(last_step["messages"][-1], "content", "")
            except Exception:
                assistant_text = ""
            self.messages = [
                *self.messages,
                {"role": "user", "content": query},
                {"role": "assistant", "content": str(assistant_text)},
            ]

    def stream(self, query: str, invoke: bool = False, remember: bool = True):
        """
        Multi-turn capable: carries prior turns stored in `self.messages`.

        - `remember=False` disables history updates (single-turn mode).
        - Use `reset_conversation()` to start a new chat.
        - `invoke=True` uses synchronous invocation instead of streaming.
        """
        # If invoke=True, use synchronous method (returns dict directly)
        if invoke:
            return self._invoke_sync(query, remember=remember)

        # Otherwise, return the generator
        return self._stream_generator(query, remember=remember)

    def run(self, query: str):
        """Run the agent on a user query (streaming) and remember conversation."""
        result = None
        for step in self.stream(query, invoke=False, remember=True):
            step["messages"][-1].pretty_print()
            result = step
        return result

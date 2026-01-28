import streamlit as st
import pandas as pd
from src.agent import RagSqlAgent


# Helper functions
def extract_reasoning(message):
    """Extract reasoning content from message content blocks."""
    if not hasattr(message, "content_blocks") or not message.content_blocks:
        return []

    reasoning_texts = []
    for block in message.content_blocks:
        if block.get("type") == "reasoning":
            reasoning_text = block.get("reasoning", "")
            if reasoning_text:
                reasoning_texts.append(reasoning_text)
    return reasoning_texts


def get_tool_status_message(tool_call):
    """Get display message for tool call."""
    tool_messages = {
        "[Tool Call] vector_search": f"Searching for: {tool_call['args'].get('query', '...')}",
        "[Tool Call] sql_db_query": "Querying database...",
        "Response": "Formatting response...",
        "give_response": "Formatting response...",
    }
    return tool_messages.get(tool_call["name"])


def parse_response_data(final_msg):
    """Extract display content and dataframe from final message."""
    display_content = getattr(final_msg, "content", "")
    response_data = None

    if hasattr(final_msg, "tool_calls") and final_msg.tool_calls:
        for tc in final_msg.tool_calls:
            if tc["name"] in ["Response", "give_response", "StructuredResponse"]:
                args = tc["args"]
                if "summary" in args:
                    display_content = args["summary"]
                if "details" in args and args["details"]:
                    response_data = pd.DataFrame(args["details"])
                break

    # Fallback to content if display_content is still empty
    if not display_content and hasattr(final_msg, "content"):
        display_content = final_msg.content

    return display_content, response_data


def should_reinitialize_agent(client_id_str, show_reasoning):
    """Check if agent needs reinitialization."""
    return (
        "agent" not in st.session_state
        or st.session_state.get("current_client_id") != client_id_str
        or st.session_state.get("current_show_reasoning") != show_reasoning
    )


def initialize_agent(client_id_str, model_name, show_reasoning):
    """Initialize or reinitialize the agent."""
    if not client_id_str.isdigit():
        st.warning("Please enter a numeric Client ID to start.")
        st.stop()

    with st.spinner("Initializing Agent..."):
        st.session_state.agent = RagSqlAgent(
            client_id=int(client_id_str),
            model_name=model_name,
            reasoning=show_reasoning,
        )
        st.session_state.current_client_id = client_id_str
        st.session_state.current_show_reasoning = show_reasoning
        st.session_state.messages = []


def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.header("Authentication")
        client_id_str = st.text_input("Client ID", value="880")

        st.header("Agent Settings")
        model_name = st.selectbox(
            "Ollama Model",
            ["qwen3:4b", "qwen3:1.7b", "gemma3:4b", "functiongemma:270m"],
            index=0,
        )

        show_reasoning = st.checkbox("Reasoning", value=True)

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            if "agent" in st.session_state and hasattr(
                st.session_state.agent, "reset_conversation"
            ):
                st.session_state.agent.reset_conversation()
            st.rerun()

    return client_id_str, model_name, show_reasoning


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_chat_history():
    """Display chat history from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Persisted collapsible reasoning (for assistant messages)
            if message.get("role") == "assistant" and message.get("reasoning"):
                with st.expander("Reasoning", expanded=False):
                    st.markdown(message["reasoning"])
            if "data" in message and message["data"] is not None:
                st.dataframe(message["data"])


def handle_chat_input(prompt):
    """Handle user chat input and agent response."""
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        status_placeholder = st.status("Thinking...")
        message_placeholder = st.empty()

        # Stream agent responses and show progress
        result = None
        reasoning_content = []

        for step in st.session_state.agent.stream(prompt):
            result = step
            final_msg = step["messages"][-1]

            # Handle reasoning updates
            new_reasoning = extract_reasoning(final_msg)
            for reasoning_text in new_reasoning:
                if reasoning_text not in reasoning_content:
                    reasoning_content.append(reasoning_text)
                    status_placeholder.markdown(
                        "**Reasoning:**\n"
                        + "\n\n".join([f"*{s.strip()}*" for s in reasoning_content])
                    )

            # Handle tool call status updates
            if hasattr(final_msg, "tool_calls") and final_msg.tool_calls:
                for tc in final_msg.tool_calls:
                    status_msg = get_tool_status_message(tc)
                    if status_msg:
                        status_placeholder.write(status_msg)

            # Handle tool completion
            if hasattr(final_msg, "type") and final_msg.type == "tool":
                status_placeholder.write(f"{final_msg.name} completed.")

        # Validate we got a response
        if not result or not result.get("messages"):
            st.error("No response received from agent.")
            st.stop()

        # Parse the final response
        final_msg = result["messages"][-1]
        display_content, response_data = parse_response_data(final_msg)

        status_placeholder.update(
            label="Analysis complete!", state="complete", expanded=False
        )

        # Display final response (reasoning is already shown in status bar)
        with message_placeholder.container():
            if display_content:
                st.markdown(display_content)

        if response_data is not None:
            st.dataframe(response_data)

        # Store in history (keep reasoning in a collapsible expander)
        reasoning_md = ""
        if reasoning_content:
            reasoning_md = "\n\n".join(
                [f"*{step.strip()}*" for step in reasoning_content]
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": display_content or "",
                "reasoning": reasoning_md,
                "data": response_data,
            }
        )


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(page_title="Financial Transaction Assistant", layout="wide")

    # Header
    st.title("Financial Transaction Assistant")
    st.markdown("""
    This assistant helps you analyze your transactions using RAG and SQL.
    Ask questions like:
    - "How much did I spend for groceries in total?"
    - "What are my latest transactions?"
    - "List my transactions in August 2023"
    """)

    # Render sidebar and get configuration
    client_id_str, model_name, show_reasoning = render_sidebar()

    # Initialize session state
    initialize_session_state()

    # Initialize or update agent if needed
    if should_reinitialize_agent(client_id_str, show_reasoning):
        initialize_agent(client_id_str, model_name, show_reasoning)

    # Display chat history
    display_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask about your transactions..."):
        handle_chat_input(prompt)


if __name__ == "__main__":
    main()

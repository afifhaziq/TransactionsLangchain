from src.agent import RagSqlAgent


def main():
    print("Initializing Local RAG-SQL Agent...")
    print("Connecting to: transactions.db")
    print("Vector Store: ./chroma_db")

    client_id = input("Enter Client ID: ").strip()
    if not client_id.isdigit():
        print("Client ID is required. Please try again.")
        return main()

    client_id = int(client_id)

    agent = RagSqlAgent(client_id=client_id)

    print("\nReady! Ask questions about your transactions (type 'exit' to quit).")
    print("Example: 'How much did I spend on groceries?'")

    while True:
        try:
            user_input = input("\nUser > ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            if not user_input.strip():
                continue

            agent.run(user_input)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()

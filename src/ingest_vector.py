import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine


def ingest_vector():
    """Ingest vector data from a SQLite database into ChromaDB
    This function extracts unique values from the 'desc', 'merchant', and 'cat' columns of the 'transactions' table in the SQLite database,
    and ingests them into a ChromaDB collection.
    """

    db_path = "transactions.db"
    collection_name = "transactions_metadata"

    print("Connecting to SQLite database...")
    engine = create_engine(f"sqlite:///{db_path}")

    # Read unique values from relevant columns
    print("Extracting unique values from columns: desc, merchant, cat...")
    try:
        df = pd.read_sql(
            "SELECT DISTINCT [desc], merchant, cat FROM transactions", engine
        )
    except Exception as e:
        print(f"Error reading from database: {e}")
        return

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")

    # Use Ollama embedding function
    print("Using Ollama embeddings (model: qwen3-embedding:0.6b)...")
    ef = embedding_functions.OllamaEmbeddingFunction(
        model_name="qwen3-embedding:0.6b",
        url="http://localhost:11434/api/embeddings",
        timeout=1000,
    )

    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}' for re-ingestion.")
    except Exception:
        pass

    collection = client.create_collection(name=collection_name, embedding_function=ef)

    # Prepare data for vector store
    documents = []
    metadatas = []
    ids = []

    print("Processing unique values...")

    # Helper to add items
    def add_items(column_name, values):
        for i, value in enumerate(values):
            if pd.isna(value) or str(value).strip() == "":
                continue

            # Use value itself as the document content for semantic search
            val_str = str(value)

            # Store metadata to know which column this value belongs to
            # We also add 'original_value' just in case
            documents.append(val_str)
            metadatas.append({"column": column_name, "original_value": val_str})
            ids.append(f"{column_name}_{i}")

    # Process each column
    add_items("desc", df["desc"].unique())
    add_items("merchant", df["merchant"].unique())
    add_items("cat", df["cat"].unique())

    print(f"Ingesting {len(documents)} items into Vector Store...")

    # Add to collection in batches
    batch_size = 5000
    total_docs = len(documents)

    for i in range(0, total_docs, batch_size):
        end = min(i + batch_size, total_docs)
        print(f"  Batch {i} to {end}...")
        collection.add(
            documents=documents[i:end], metadatas=metadatas[i:end], ids=ids[i:end]
        )

    print("Vector ingestion complete.")


if __name__ == "__main__":
    ingest_vector()

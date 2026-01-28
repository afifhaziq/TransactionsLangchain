import pandas as pd
from sqlalchemy import create_engine
import re


def clean_column_name(name):
    """Clean column name to be SQL friendly."""
    # Remove special characters and spaces, convert to snake_case
    clean = re.sub(r"[^a-zA-Z0-9]", "_", name.strip())
    # Remove multiple underscores
    clean = re.sub(r"_+", "_", clean)
    # Remove trailing/leading underscores
    clean = clean.strip("_")
    return clean.lower()


def ingest_sql():
    csv_path = "data.csv"
    db_path = "transactions.db"

    print(f"Reading {csv_path}...")

    # Load data
    df = pd.read_csv(csv_path)

    # Clean column names
    print("Cleaning column names...")
    original_columns = df.columns.tolist()
    df.columns = [clean_column_name(col) for col in df.columns]

    print(f"Original columns: {original_columns}")
    print(f"New columns:      {df.columns.tolist()}")

    # Create SQLite engine
    print(f"Connecting to SQLite database at {db_path}...")
    engine = create_engine(f"sqlite:///{db_path}")

    print("Writing data to 'transactions' table...")

    try:
        df.to_sql("transactions", engine, if_exists="replace", index=False)
        print(f"Successfully loaded {len(df)} rows into transactions.db")
    except Exception as e:
        print(f"Error writing to database: {e}")


if __name__ == "__main__":
    ingest_sql()

"""
clean_excel.py
--------------
Run this script once to clean your MTDR Excel and index it into ChromaDB.
Usage: python clean_excel.py --file "MTDR Records.xlsx"
"""

import argparse
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ── Config ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "mttr_records"
DB_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Exact column names from your Excel
COL_SMD_LINE = "SMD Line"
COL_MACHINE  = "Machine Type"
COL_PROBLEM  = "Problem"
COL_SOLUTION = "Solution"


def clean_text(val):
    if pd.isna(val):
        return ""
    text = str(val).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\u0900-\u097F]", "", text)
    return text.strip()


def load_and_clean(filepath: str) -> pd.DataFrame:
    print(f"[1/3] Reading: {filepath}")
    df = pd.read_excel(filepath, engine="openpyxl")

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # ✅ Strict column check
    required_cols = [COL_SMD_LINE, COL_MACHINE, COL_PROBLEM, COL_SOLUTION]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing column: '{col}'\nAvailable columns: {list(df.columns)}"
            )

    print("   Columns verified ✅")

    cleaned = pd.DataFrame({
        "smd_line": df[COL_SMD_LINE].apply(clean_text),
        "machine":  df[COL_MACHINE].apply(clean_text),
        "problem":  df[COL_PROBLEM].apply(clean_text),
        "solution": df[COL_SOLUTION].apply(clean_text),
    })

    before = len(cleaned)

    cleaned = cleaned[
        (cleaned["problem"].str.len() > 5) &
        (cleaned["solution"].str.len() > 5)
    ].drop_duplicates(subset=["smd_line", "machine", "problem", "solution"])

    print(f"   Rows: {before} → {len(cleaned)} after cleaning")

    return cleaned.reset_index(drop=True)


def index_to_chromadb(df: pd.DataFrame):
    print(f"[2/3] Connecting to ChromaDB at '{DB_PATH}'")
    client = chromadb.PersistentClient(path=DB_PATH)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="./local_model" #changes EMBED_MODEL to local path
    )

    # Delete old collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print("   Deleted existing collection (rebuilding fresh)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():
        doc = f"SMD Line: {row['smd_line']}. Machine: {row['machine']}. Problem: {row['problem']}"
        
        documents.append(doc)
        metadatas.append({
            "smd_line": row["smd_line"],
            "machine":  row["machine"],
            "problem":  row["problem"],
            "solution": row["solution"],
        })
        ids.append(f"rec_{i}")

    # Batch insert
    batch = 500
    for start in range(0, len(documents), batch):
        collection.add(
            documents=documents[start:start+batch],
            metadatas=metadatas[start:start+batch],
            ids=ids[start:start+batch],
        )
        print(f"   Indexed {min(start+batch, len(documents))}/{len(documents)} records")

    print(f"[3/3] Done. {len(documents)} records indexed.")


def main():
    parser = argparse.ArgumentParser(description="Clean MTTR Excel and index into ChromaDB")
    parser.add_argument("--file", required=True, help="Path to your MTTR Excel file")
    args = parser.parse_args()

    df = load_and_clean(args.file)
    index_to_chromadb(df)

    print("\n✅ All done!")
    print("Run your backend with: uvicorn app:app --host 127.0.0.1 --port 8000")


if __name__ == "__main__":
    main()

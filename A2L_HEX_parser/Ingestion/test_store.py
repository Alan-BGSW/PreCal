"""Inspect the local store after ingesting.

    python verify_store.py
"""

import os
import re
import chromadb


def _default_collection_name() -> str:
    configured = os.getenv("COLLECTION_NAME")
    if configured:
        return configured
    backend = os.getenv("EMBED_BACKEND", "local").strip().lower()
    if backend == "local":
        model = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()
        return f"labels_local_{safe[:40]}"
    return "labels"


def _resolve_collection_name(client: chromadb.PersistentClient, preferred_name: str) -> str:
    existing = [c.name for c in client.list_collections()]
    if preferred_name in existing:
        return preferred_name

    if not existing:
        raise SystemExit(
            "No collections found in ./label_store. Run ingestion first or point to the correct persist directory."
        )

    if len(existing) == 1:
        only = existing[0]
        print(
            f"[warn] Preferred collection '{preferred_name}' not found; using only available collection '{only}'."
        )
        return only

    print(f"[error] Preferred collection '{preferred_name}' not found.")
    print("Available collections:")
    for name in existing:
        print(f"  - {name}")
    raise SystemExit(
        "Set COLLECTION_NAME to one of the above names and rerun."
    )


client = chromadb.PersistentClient(path="./label_store")
collection_name = _default_collection_name()
collection_name = _resolve_collection_name(client, collection_name)
col = client.get_collection(collection_name)

print("collection:", collection_name)
print("total records:", col.count())

# Peek at a few stored records + their metadata.
sample = col.get(limit=5, include=["metadatas", "documents"])
for meta, doc in zip(sample["metadatas"], sample["documents"]):
    print("\n--", meta["label_path"])
    print("   fn:", meta["normalized_function_name"],
          "| ver:", meta["version"], "| role:", meta["doc_role"])

# The core operation: candidates for an OLD label, scoped to its function,
# restricted to NEW labels only. (Embed the query with input_type='query'.)
print("\n=== filtered query: new labels in elm_engine ===")
res = col.get(
    where={"$and": [
        {"normalized_function_name": "elm_engine"},
        {"doc_role": "new"},
    ]},
    include=["metadatas"],
)
for meta in res["metadatas"]:
    print("  ", meta["label_path"])
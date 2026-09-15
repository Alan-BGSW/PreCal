"""Lookup function name(s) for a label from Chroma metadata.

Example:
    .\\.venv\\Scripts\\python.exe .\\lookup_function_by_label.py "CoElM_SM01_I.CoElM_StMWaitTOnD_I"
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import chromadb
from chromadb.errors import NotFoundError


def _resolve_collection_name(client: chromadb.PersistentClient, preferred_name: str) -> str:
    existing = [c.name for c in client.list_collections()]
    if preferred_name in existing:
        return preferred_name

    if not existing:
        raise RuntimeError(
            "No Chroma collections found in the selected persist directory. "
            "Run ingestion first or pass --persist-dir to the correct store."
        )

    if len(existing) == 1:
        only = existing[0]
        print(
            f"[warn] Preferred collection '{preferred_name}' not found; using only available collection '{only}'."
        )
        return only

    names = ", ".join(existing)
    raise RuntimeError(
        f"Preferred collection '{preferred_name}' not found. "
        f"Available collections: {names}. "
        "Pass --collection-name explicitly."
    )


def _default_collection_name() -> str:
    # Kept local to avoid importing heavy embedding modules for this metadata-only utility.
    import os
    import re

    configured = os.getenv("COLLECTION_NAME")
    if configured:
        return configured

    backend = os.getenv("EMBED_BACKEND", "local").strip().lower()
    if backend == "local":
        model = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()
        return f"labels_local_{safe[:40]}"
    return "labels"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lookup metadata function name(s) for a label path in Chroma."
    )
    parser.add_argument("label", help="Exact label_path to search for.")
    parser.add_argument("--persist-dir", default="./label_store", help="Chroma persistent path.")
    parser.add_argument("--collection-name", default=None, help="Override Chroma collection name.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Maximum matching chunks to scan for aggregation (default: 100000).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    label = args.label.strip()
    if not label:
        raise SystemExit("label must not be empty")

    client = chromadb.PersistentClient(path=args.persist_dir)
    preferred_name = args.collection_name or _default_collection_name()
    name = _resolve_collection_name(client, preferred_name)

    try:
        col = client.get_collection(name)
    except NotFoundError as exc:
        raise RuntimeError(
            f"Collection '{name}' does not exist in persist_dir='{args.persist_dir}'."
        ) from exc

    result = col.get(
        where={"label_path": label},
        include=["metadatas"],
        limit=args.limit,
    )

    metas = result.get("metadatas") or []
    if not metas:
        print(f"No exact metadata match found for label_path: {label}")
        print("Tip: check spelling/case or try providing --collection-name explicitly.")
        return 2

    per_function: dict[str, dict[str, set[str] | int]] = defaultdict(
        lambda: {"roles": set(), "versions": set(), "count": 0}
    )
    for meta in metas:
        fn = str(meta.get("normalized_function_name") or "<missing>")
        role = str(meta.get("doc_role") or "<missing>")
        ver = str(meta.get("version") or "<missing>")

        entry = per_function[fn]
        entry["count"] = int(entry["count"]) + 1
        cast_roles = entry["roles"]
        cast_versions = entry["versions"]
        assert isinstance(cast_roles, set)
        assert isinstance(cast_versions, set)
        cast_roles.add(role)
        cast_versions.add(ver)

    print(f"Label: {label}")
    print(f"Collection: {name}")
    print(f"Matching chunks: {len(metas)}")
    print("Function name candidates:")

    sorted_items = sorted(
        per_function.items(),
        key=lambda kv: (-int(kv[1]["count"]), kv[0]),
    )
    for fn, info in sorted_items:
        roles = sorted(cast for cast in info["roles"] if isinstance(cast, str))
        versions = sorted(cast for cast in info["versions"] if isinstance(cast, str))
        count = int(info["count"])
        print(f"- {fn} | chunks={count} | roles={','.join(roles)} | versions={','.join(versions)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

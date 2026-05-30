"""Retrieval helpers for day 3.

This module reads from the persistent Chroma collection created during
indexing and returns the most relevant chunks for a text query.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    distance: float | None = None


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Create the local sentence-transformers embedder lazily."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_chroma_collection(chroma_dir: str | Path, collection_name: str = DEFAULT_COLLECTION_NAME):
    """Open the persistent Chroma collection used by the indexer."""

    import chromadb

    client = chromadb.PersistentClient(path=str(Path(chroma_dir)))
    return client.get_or_create_collection(name=collection_name)


def search(query: str, chroma_dir: str | Path = "data/chroma", k: int = 3, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[RetrievedChunk]:
    """Return the most relevant chunks for a query."""

    if not query.strip():
        return []

    collection = get_chroma_collection(chroma_dir)
    embedder = get_embedder(model_name)
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[RetrievedChunk] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        retrieved.append(
            RetrievedChunk(
                text=document,
                source_file=str(metadata.get("source_file", "unknown")),
                page_number=int(metadata.get("page_number", 0)),
                chunk_index=int(metadata.get("chunk_index", 0)),
                distance=float(distance) if distance is not None else None,
            )
        )

    return retrieved


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    """Format retrieval output for quick inspection in the terminal."""

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{index}] {chunk.source_file} p.{chunk.page_number} c.{chunk.chunk_index} "
            f"distance={chunk.distance}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build a small CLI for manual retrieval tests."""

    parser = argparse.ArgumentParser(description="Search the local Chroma index")
    parser.add_argument("query")
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    return parser


def main() -> None:
    """Run a one-off retrieval from the command line."""

    args = build_parser().parse_args()
    chunks = search(
        query=args.query,
        chroma_dir=args.chroma_dir,
        k=args.top_k,
        model_name=args.model_name,
    )
    print(format_retrieved_chunks(chunks))


if __name__ == "__main__":
    main()

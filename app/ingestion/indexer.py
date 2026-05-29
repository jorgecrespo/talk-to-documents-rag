"""Chroma indexing utilities.

This module builds a persistent local vector store from PDFs that were
already loaded and chunked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import Chunk, chunk_pages, save_chunks_to_json
from .loader import load_pdfs_from_dir, save_pages_to_json


DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Create the local sentence-transformers embedder lazily."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_chroma_collection(chroma_dir: str | Path, collection_name: str = DEFAULT_COLLECTION_NAME):
    """Create or open a persistent Chroma collection."""

    import chromadb

    client = chromadb.PersistentClient(path=str(Path(chroma_dir)))
    return client.get_or_create_collection(name=collection_name)


def build_chunk_id(chunk: Chunk) -> str:
    """Build a stable id so reindexing updates existing records."""

    return f"{chunk.source_file}::p{chunk.page_number}::c{chunk.chunk_index}"


def index_chunks(chunks: list[Chunk], collection, embedder) -> int:
    """Embed and upsert chunks into Chroma."""

    if not chunks:
        return 0

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
    ids = [build_chunk_id(chunk) for chunk in chunks]
    metadatas = [
        {
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def index_pdfs(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    chroma_dir: str | Path = "data/chroma",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> int:
    """Load PDFs, chunk them, persist intermediates, and store embeddings."""

    pages = load_pdfs_from_dir(raw_dir)
    if not pages:
        raise FileNotFoundError(f"No PDFs with extractable text found in {raw_dir}")

    processed_path = Path(processed_dir)
    save_pages_to_json(pages, processed_path / "pages.json")

    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    save_chunks_to_json(chunks, processed_path / "chunks.json")

    collection = get_chroma_collection(chroma_dir, collection_name=collection_name)
    embedder = get_embedder(model_name)
    return index_chunks(chunks, collection, embedder)


def build_parser() -> argparse.ArgumentParser:
    """Build the small CLI used to run day 2 manually."""

    parser = argparse.ArgumentParser(description="Index PDFs into local Chroma")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    return parser


def main() -> None:
    """Run the indexer from the command line."""

    args = build_parser().parse_args()
    indexed = index_pdfs(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Indexed {indexed} chunks into Chroma.")


if __name__ == "__main__":
    main()

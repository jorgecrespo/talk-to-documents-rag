"""Question answering over retrieved document chunks.

This module stays intentionally small for the project V1: it takes the
retrieved chunks from the retriever module, builds a prompt, and asks OpenAI
for an answer with source-aware grounding.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Sequence

from app.retrieval.retriever import RetrievedChunk, search


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[RetrievedChunk]


def format_sources(chunks: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks into a compact context block."""

    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"Source {index}: {chunk.source_file} | page {chunk.page_number} | chunk {chunk.chunk_index}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
    """Build the chat messages for the OpenAI request."""

    system_prompt = (
        "You answer questions only from the provided document context. "
        "If the context does not contain the answer, say you do not know. "
        "Cite the source file and page number when relevant."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{format_sources(chunks)}\n\n"
        "Answer in a concise but helpful way."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def get_openai_client():
    """Create the OpenAI client lazily after loading environment variables."""

    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def answer_question(
    question: str,
    chunks: Sequence[RetrievedChunk],
    model_name: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0,
) -> AnswerResult:
    """Generate an answer grounded in the retrieved chunks."""

    if not question.strip():
        raise ValueError("question cannot be empty")

    client = get_openai_client()
    messages = build_messages(question, chunks)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )

    answer = response.choices[0].message.content or ""
    return AnswerResult(answer=answer.strip(), sources=list(chunks))


def answer_from_query(
    question: str,
    chroma_dir: str = "data/chroma",
    top_k: int = 3,
    retrieval_model_name: str = "BAAI/bge-small-en-v1.5",
    openai_model_name: str = DEFAULT_OPENAI_MODEL,
) -> AnswerResult:
    """Retrieve context first, then ask OpenAI for the final answer."""

    chunks = search(
        query=question,
        chroma_dir=chroma_dir,
        k=top_k,
        model_name=retrieval_model_name,
    )
    return answer_question(question, chunks, model_name=openai_model_name)


def build_parser() -> argparse.ArgumentParser:
    """Build a tiny CLI for one-off QA tests."""

    parser = argparse.ArgumentParser(description="Answer questions from indexed documents")
    parser.add_argument("question")
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retrieval-model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--openai-model-name", default=DEFAULT_OPENAI_MODEL)
    return parser


def main() -> None:
    """Run a one-off QA query from the command line."""

    args = build_parser().parse_args()
    result = answer_from_query(
        question=args.question,
        chroma_dir=args.chroma_dir,
        top_k=args.top_k,
        retrieval_model_name=args.retrieval_model_name,
        openai_model_name=args.openai_model_name,
    )
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"- {source.source_file} p.{source.page_number} c.{source.chunk_index}")


if __name__ == "__main__":
    main()

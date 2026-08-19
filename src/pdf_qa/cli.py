"""CLI entry point: `pdf-qa <path-to-pdf>` extracts the PDF, builds a
search index, and drops you into a Q&A loop.

This wires the pipeline together, but the pipeline is only half-built:
chunking.chunk_pages() and agent.ask() are TODOs (see their docstrings)
and will raise AttributeError until you implement them. The call
signatures below are a starting point, not a contract -- change them
however your agent.py design needs.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from dotenv import load_dotenv

from pdf_qa import agent, chunking, embeddings, extraction


def build_index(pdf_path: str) -> tuple[list, np.ndarray]:
    """Extract, chunk, and embed a PDF once up front.

    Embedding every chunk eagerly at startup (rather than lazily, only
    for chunks the agent's search actually touches) is the simplest
    design and the one wired in here -- cheap at this scale. Revisit if
    you start feeding in much larger documents.

    Vectors are normalized to unit length here, once, so search.py's
    search_pdf() can score a query against the whole corpus with a
    single dot product instead of re-normalizing every chunk on every
    question.
    """
    pages = extraction.extract_pages(pdf_path)
    chunks = chunking.chunk_pages(pages)
    vectors = np.array(embeddings.embed_texts([chunk.text for chunk in chunks]))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return chunks, vectors


def qa_loop(pdf_path: str) -> None:
    print(f"Indexing {pdf_path}...")
    chunks, vectors = build_index(pdf_path)
    print(f"Ready. {len(chunks)} chunks indexed. Ask a question (Ctrl+D to quit).\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue

        answer = agent.ask(question, chunks, vectors)  # TODO: implement in agent.py
        print(f"\n{answer}\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Agentic PDF Q&A")
    parser.add_argument("pdf_path", help="Path to the PDF to ask questions about")
    parser.add_argument(
        "--mode",
        choices=["rag", "stuff"],
        default="rag",
        help="rag: agentic search over chunks (default). "
        "stuff: whole PDF in context, no retrieval -- TODO, see stuff.py.",
    )
    args = parser.parse_args()

    if args.mode == "stuff":
        print("--mode stuff is not implemented yet -- see stuff.py.", file=sys.stderr)
        sys.exit(1)

    qa_loop(args.pdf_path)


if __name__ == "__main__":
    main()

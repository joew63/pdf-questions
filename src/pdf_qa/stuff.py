""" "Stuff the whole PDF into context" mode -- no retrieval, no chunking.

TODO(you): implement this after the RAG path (agent.py) works, as a
cost/quality comparison. Concatenate every page's text from
extraction.extract_pages() into one big context block, send it to Claude
in a single call_llm() call alongside the question, and return the
answer. No search_pdf tool, no embeddings, no agent loop.

Do not build this yet -- it's here as a placeholder so the CLI can
eventually route to it via a --mode flag (see cli.py).
"""

from __future__ import annotations

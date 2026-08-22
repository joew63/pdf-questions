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
from pdf_qa.extraction import Page
from pdf_qa.llm import call_llm
from pdf_qa.agent import extraction

SYSTEM_PROMPT = """
Read the page_text's reference material before answering. The question to be answered is located last, after the reference material. Cite pages where answer was found in the answer.
"""

def ask(question: str, pages: list[Page]) -> str:
    page_text = "Reference Material: "
    for page in pages:
        page_text += f"[p. {page.number}]\n{page.text}\n\n"
    page_text += f"\nQuestion: {question}"
    messages: list[dict] = [{"role": "user", "content": page_text}]

    response = call_llm(messages, system=SYSTEM_PROMPT)
    return extraction(response)
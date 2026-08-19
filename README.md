# pdf-qa

Ask questions about a PDF. Instead of a fixed retrieve-then-generate
pipeline, Claude gets a `search_pdf` tool and decides for itself whether
and how many times to call it before answering.

## Setup

```bash
pip install -e .
cp .env.example .env
```

Fill in `.env` with an [Anthropic API key](https://console.anthropic.com/settings/keys) (the LLM) and an [OpenAI API key](https://platform.openai.com/api-keys) (embeddings only).

## Usage

```bash
pdf-qa path/to/your.pdf
```

Drops you into a `>` prompt. Ask questions, `Ctrl+D` to quit.

## How it works

1. `extraction.py` pulls text out of the PDF, page by page.
2. `chunking.py` splits each page into overlapping chunks.
3. `embeddings.py` embeds every chunk once, up front.
4. You ask a question. Claude (`agent.py`) decides whether to call
   `search_pdf` (`search.py`), which does a cosine-similarity lookup over
   the chunk embeddings and returns the most relevant passages, page
   numbers included.
5. Claude answers, citing the pages it used. It can call `search_pdf`
   more than once per question if it needs to.

## Project layout

```
src/pdf_qa/
├── extraction.py   PDF -> per-page text (PyMuPDF)
├── chunking.py     page text -> overlapping chunks
├── embeddings.py   OpenAI embedding wrapper
├── search.py       search_pdf tool: cosine similarity over chunk embeddings
├── llm.py          Claude Messages API wrapper
├── agent.py        the tool-use loop -- decides when to search, builds the final answer
├── stuff.py        whole-PDF-in-context mode, for a cost/quality comparison -- not built yet
└── cli.py          `pdf-qa <pdf>` entry point
```

## Status

RAG mode (the default) works end to end. `--mode stuff` is stubbed but not implemented.

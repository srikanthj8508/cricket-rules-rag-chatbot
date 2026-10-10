# cricket-rules-rag-chatbot

A retrieval-augmented chatbot that answers questions about cricket's playing conditions and
regulations from ICC documents (icc-cricket.com), cites a source on every sentence, and refuses when the
retrieved context is insufficient. Unofficial student project; not affiliated
with the ICC or MCC.

## Pipeline

| Stage | Module | Command |
| --- | --- | --- |
| Scrape | `cricketrag/crawl/spider.py` | `python -m cricketrag.crawl.spider --limit 50` |
| Extraction | `cricketrag/extract/extract.py` | `python -m cricketrag.extract.extract` |
| Catalog and manifest | `cricketrag/catalog.py` | `python -m cricketrag.catalog` (also run by the index build) |
| Chunking | `cricketrag/chunk/chunker.py` | (called by the index build) |
| Embedding + vector DB | `cricketrag/index/build.py` | `python -m cricketrag.index.build --strategy struct` |
| Retrieval + reranking | `cricketrag/retrieve/hybrid.py` | |
| Generation, citations, refusal | `cricketrag/generate/answer.py` | |
| Chat interface | `app/streamlit_app.py` | `streamlit run app/streamlit_app.py` |
| Evaluation | `eval/run.py` | `python -m eval.run --split dev` |

## Setup

These setup commands require the other team modules to be merged. This backend
contribution supplies configuration, dependencies and the launch script; it
does not include the crawler, retriever, generator, interface or tests.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY, then: set -a; source .env; set +a
pytest -q
```

The crawler identifies the project using `CRICKET_CONTACT_EMAIL`, configurable
in `.env` (default: srikanthreddy8508@gmail.com). Keep real API keys in `.env`,
which is git-ignored. `run.sh` loads `.env` before any pipeline stage.

`run.sh` is for Bash on macOS/Linux or Windows through WSL/Git Bash. On native
Windows PowerShell, activate the virtual environment and run the stage commands
in the pipeline table individually; set API credentials in the environment first.

## Re-chunking

Change a parameter, bump its entry in `STRATEGY_VERSIONS` in `config.py`, and
re-run the index build. It reads `data/extracted/` only (no re-crawl), writes
the new version beside the old one, checks coverage, flips `active_version`,
then deletes every chunk from other versions. Pass `--keep-old` while
evaluating both strategies side by side.

## Crawl etiquette

robots.txt obeyed, one request at a time, 2 s delay with auto-throttle,
identifying User-Agent, HTTP cache so re-runs do not re-fetch. No logins,
forms, or search endpoints. ICC documents are copyrighted: the raw and
extracted text stays in the git-ignored `data/` folder and is never
redistributed.

## How a question is answered

1. The question is tidied by the LLM: spelling fixed, fragments completed, and
   comparisons split into one short search per part.
2. Dense (bge-small) and BM25 results are fused with reciprocal rank fusion.
   Editions replaced by a newer one are skipped unless the question names a year.
3. A cross-encoder reranks the candidates; near-identical clauses from
   different documents are collapsed so each slot holds different evidence.
4. If the best score is below the threshold, the bot refuses.
5. The LLM answers from the numbered passages only. Every sentence must carry a
   citation; a failing draft is repaired once, then trimmed to its cited
   sentences, and refused if none remain.

Without `ANTHROPIC_API_KEY` the app uses a local Ollama model if one is running
(`ollama pull llama3.2`), otherwise it shows the retrieved passages only.

## Reported corpus (team crawl dated 2026-10-10)

378 documents fetched (102 HTML pages, 276 PDFs); 377 extracted. 245 are in
scope after excluding named disciplinary decisions, off-topic publications and
documents with no extractable text. `data/manifest.csv` lists every document
with its title, edition year, table count and exclusion reason. Source text is
copyrighted by the ICC and is not committed. These figures are reported by the
team; this backend ZIP contains no corpus or manifest to verify them.

## Status

The team reports an end-to-end run and 19 passing unit tests in the assembled
project. Neither result has been independently verified for this backend-only
branch. The separately shared evaluation part contains 41 draft questions;
human verification and the evaluation results still need confirmation.

## AI assistance

Code and documentation in this repository were written with the help of
Claude (Anthropic).

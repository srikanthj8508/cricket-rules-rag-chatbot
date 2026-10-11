#!/bin/bash
# One-command setup and launch. Usage:  ./run.sh        (first run: ~60 pages)
#                                       LIMIT=900 FULL=1 ./run.sh   (full crawl incl. news)
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p data/logs
PY=${PYTHON:-python3}
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "== 1/6 installing requirements (first time takes several minutes)"
pip install -q --upgrade pip 2>&1 | tee data/logs/install.log
pip install -q -r requirements.txt 2>&1 | tee -a data/logs/install.log
echo "== 2/6 unit tests"
pytest -q 2>&1 | tee data/logs/tests.log
echo "== 3/6 crawling icc-cricket.com (2 s between requests)"
if [ -n "${FULL:-}" ]; then
    unset CRICKET_SKIP_NEWS
else
    export CRICKET_SKIP_NEWS=1
fi
python -m cricketrag.crawl.spider --limit "${LIMIT:-60}" 2>&1 | tee data/logs/crawl.log
echo "== 4/6 extracting text and tables"
python -m cricketrag.extract.extract 2>&1 | tee data/logs/extract.log
echo "== 5/6 chunking, embedding, indexing (downloads the embedding model once)"
python -m cricketrag.index.build --strategy struct 2>&1 | tee data/logs/index.log
echo "== 6/6 starting the chat app"
[ -n "${ANTHROPIC_API_KEY:-}" ] || echo "No ANTHROPIC_API_KEY: use local Ollama if available, otherwise retrieval-only mode."
streamlit run app/streamlit_app.py 2>&1 | tee data/logs/app.log

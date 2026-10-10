"""Stage 8: grounded generation with per-sentence citations and three refusal gates."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, timezone

from cricketrag import config

SYSTEM = f"""You answer questions about cricket's playing conditions, regulations and governance using ONLY the numbered sources provided, which come from International Cricket Council (ICC) documents.
Rules:
- End every factual sentence with one or more citations like [1] or [2][3].
- Use only source numbers that were provided. Never use outside knowledge.
- If the sources answer only part of the question, answer that part with citations, then add one final sentence beginning "Not covered:" that names what the sources do not say.
- Only if the sources contain nothing that helps, reply with exactly: {config.REFUSAL_TOKEN}
- Rules differ by format (Test, ODI, T20I) and by men's and women's cricket. Say which format or document each rule comes from, and do not apply one format's rule to another.
- Sources can come from different editions. Prefer the most recent edition unless the question names a year, and state which edition you used.
- Do not give betting tips, match predictions, or opinions on players or umpiring decisions.
- Write plain sentences in one or two short paragraphs. No headings, no bullet lists, no bold.
- Put the citation inside the sentence, before its full stop, like this [1].
- Be concise: at most 6 sentences."""

REWRITE_SYSTEM = """You tidy up questions for a cricket rules search engine.
Rewrite the user's message as one clear, correctly spelled question.
- Fix spelling mistakes and run-together words (for example "noball" -> "no ball", "wickit" -> "wicket").
- Recognise misspelt cricket terms and write the standard term (for example "ode" or "odi" -> "ODI", "t20" -> "T20I", "drs" -> "DRS", "lwb" -> "LBW").
- Turn fragments into full questions (for example "no ball?" -> "What is a no ball and when is it called?").
- Keep the user's meaning. Do not answer the question and do not add facts.
Line 1 of your reply is the rewritten question, and nothing else.
If the question compares formats or has several separate parts, add one more line per part, each starting with "SEARCH: " and holding a short keyword query for that part alone (for example "SEARCH: ODI powerplay overs fielding restrictions" and "SEARCH: T20I powerplay overs fielding restrictions"). At most three SEARCH lines. A simple question gets none."""


def plan_query(question: str, llm) -> tuple[str, list[str]]:
    """(tidied question, extra sub-queries for comparisons). Falls back to the original."""
    try:
        lines = [l.strip().strip('"') for l in llm(REWRITE_SYSTEM, question).strip().splitlines() if l.strip()]
    except Exception:
        return question, []
    if not lines or not 3 <= len(lines[0]) <= 300 or lines[0].upper().startswith("SEARCH:"):
        return question, []
    subs = [l[7:].strip() for l in lines[1:] if l.upper().startswith("SEARCH:")]
    return lines[0], [s for s in subs if 3 <= len(s) <= 200][:3]


def rewrite_query(question: str, llm) -> str:
    """Spelling- and fragment-tolerant search query. Falls back to the original on any problem."""
    return plan_query(question, llm)[0]


REFUSAL_TEXT = ("I can't answer that from the ICC documents I have indexed. "
                "The closest pages are listed below. For anything they don't cover, "
                "check icc-cricket.com or the MCC Laws of Cricket directly.")

_PII = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b\d{2}-\d{7}\b"), "[EIN]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[PHONE]"),
]
_CITE = re.compile(r"\[(\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def mask_pii(text: str) -> tuple[str, bool]:
    found = False
    for pattern, label in _PII:
        text, n = pattern.subn(label, text)
        found = found or n > 0
    return text, found


_ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|no|cl|para|approx|inc)\.", re.I)


def _units(answer: str) -> list[str]:
    """Split an answer into the sentences that each need a citation."""
    text = re.sub(r"([.!?])\s*((?:\[\d+\])+)", r" \2\1", answer)   # "... rule. [1]" -> "... rule [1]."
    text = _ABBREV.sub(lambda m: m.group(1) + "\u2024", text)        # keep "e.g." from ending a sentence
    units = []
    for line in text.splitlines():                                   # a bullet is its own unit
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line).strip()
        units += [s.strip() for s in _SENT.split(line) if s.strip()]
    return units


def check_citations(answer: str, n_sources: int) -> list[str]:
    """Returns a list of problems; empty means every factual sentence is validly cited."""
    problems = []
    for sent in _units(answer):
        plain = sent.strip("*_# ")
        if (len(re.findall(r"[A-Za-z]{2,}", plain)) < 3      # fragments and labels
                or plain.endswith(":")                        # a lead-in to a list
                or plain.startswith("Not covered:")):         # the statement of what is missing
            continue
        cites = [int(c) for c in _CITE.findall(sent)]
        if not cites:
            problems.append(f"uncited: {sent[:60]}")
        elif any(c < 1 or c > n_sources for c in cites):
            problems.append(f"bad citation number: {sent[:60]}")
    return problems


def keep_cited(answer: str, n_sources: int) -> str:
    """The answer with every uncited or wrongly cited sentence removed ('' if nothing is left)."""
    kept = []
    for sent in _units(answer):
        cites = [int(c) for c in _CITE.findall(sent)]
        if sent.startswith("Not covered:") or (cites and all(1 <= c <= n_sources for c in cites)):
            kept.append(sent.replace("\u2024", "."))
    return " ".join(kept) if any(_CITE.search(s) for s in kept) else ""


def build_prompt(question: str, chunks: list[dict]) -> str:
    sources = []
    for i, c in enumerate(chunks, 1):
        loc = f", page {c['page']}" if c.get("page") else ""
        tags = ", ".join(str(t) for t in (c.get("gender"), c.get("format"),
                                          c.get("edition_year") and f"edition {c['edition_year']}") if t)
        sources.append(f"[{i}] {c['title']}{' (' + tags + ')' if tags else ''}{loc} ({c['url']})\n{c['text']}")
    return "Sources:\n\n" + "\n\n".join(sources) + f"\n\nQuestion: {question}"


def llm_backend() -> str:
    """'anthropic' if a key is set, 'ollama' if a local Ollama server answers, else ''."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    try:
        with urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=2) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        return "ollama" if any(n.split(":")[0] == config.OLLAMA_MODEL.split(":")[0] for n in names) else ""
    except Exception:
        return ""


def call_ollama(system: str, user: str) -> str:
    body = json.dumps({
        "model": config.OLLAMA_MODEL, "stream": False,
        "options": {"temperature": 0, "num_ctx": config.OLLAMA_CONTEXT, "num_predict": 600},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(config.OLLAMA_URL + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"].strip()


def call_llm(system: str, user: str) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return call_ollama(system, user)
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(model=config.LLM_MODEL, max_tokens=600, temperature=0,
                                 system=system, messages=[{"role": "user", "content": user}])
    return msg.content[0].text.strip()


def answer(question: str, retriever, llm=call_llm, use_gates: bool = True, rewrite: bool = True) -> dict:
    """Full online path. Returns {answer, refused, reason, sources, pii_masked, search_query}."""
    question, pii = mask_pii(question)
    search_query, subs = plan_query(question, llm) if (rewrite and retriever) else (question, [])
    if subs and hasattr(retriever, "search_multi"):
        chunks = retriever.search_multi(search_query, subs)
    else:
        chunks = retriever.search(search_query) if retriever else []
    result = {"question": question, "search_query": search_query, "sub_queries": subs, "sources": chunks,
              "pii_masked": pii, "refused": False, "reason": None}
    question = search_query  # the model answers the tidied question; the log keeps the original

    def refuse(reason: str) -> dict:
        result.update(answer=REFUSAL_TEXT, refused=True, reason=reason)
        return _log(result)

    # Gate 1: retrieval confidence, before spending an LLM call
    top = chunks[0].get("score") if chunks else None
    if use_gates and retriever and (not chunks or (top is not None and top < config.REFUSAL_THRESHOLD)):
        return refuse("low_retrieval_score")

    prompt = build_prompt(question, chunks) if chunks else f"Question: {question}"
    system = SYSTEM if chunks else "Answer the question about cricket rules concisely."  # B0
    text = llm(system, prompt)
    # Gate 2: the model itself says the context is insufficient
    if config.REFUSAL_TOKEN in text:
        return refuse("model_insufficient_context")
    if chunks and use_gates:
        # Gate 3: every factual sentence must carry a valid citation
        problems = check_citations(text, len(chunks))
        if problems:
            result["draft"], result["problems"] = text, problems  # kept in the log for diagnosis
            # one regeneration: the model repairs its own draft
            retry = llm(system, prompt + "\n\nYour draft answer:\n" + text + "\n\nThese sentences have no "
                        "valid citation:\n- " + "\n- ".join(problems) + "\n\nRewrite the draft. Give every "
                        "sentence a citation [n] to a source that supports it, and delete any sentence no "
                        "source supports. Keep everything that was correctly cited.")
            if config.REFUSAL_TOKEN not in retry and not check_citations(retry, len(chunks)):
                text = retry
            else:
                # last resort: show only the sentences that are cited; refuse if none are
                text = keep_cited(text, len(chunks))
                if not text:
                    return refuse("citation_check_failed")
                result["trimmed"] = True
    result["answer"] = text
    return _log(result)


def _log(result: dict) -> dict:
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "question": result["question"],
           "search_query": result.get("search_query"), "sub_queries": result.get("sub_queries"),
           "draft": result.get("draft") if result["refused"] else None,
           "problems": result.get("problems") if result["refused"] else None,
           "chunk_ids": [c["chunk_id"] for c in result["sources"]],
           "scores": [c.get("score") for c in result["sources"]],
           "answer": result["answer"], "refused": result["refused"], "reason": result["reason"]}
    with config.LOG_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return result

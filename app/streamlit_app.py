"""Stage 9: chat interface.  Run:  streamlit run app/streamlit_app.py"""
import os
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cricketrag import config  # noqa: E402
from cricketrag.generate.answer import answer, llm_backend  # noqa: E402
from cricketrag.retrieve.hybrid import Retriever  # noqa: E402

st.set_page_config(page_title="Cricket Rules (student project)")
st.title("Cricket Rules Chatbot")
st.caption("Unofficial student project. Not affiliated with the ICC or MCC. "
           "Answers come only from ICC playing conditions and regulations and cite their source. "
           "No betting advice or match predictions.")


BACKEND = llm_backend()
HAS_KEY = bool(BACKEND)
if BACKEND == "ollama":
    st.caption(f"Answers written by a local model ({config.OLLAMA_MODEL}) running on this computer.")
elif not BACKEND:
    st.warning("No language model found, so answers are switched off. Start Ollama and run "
               f"`ollama pull {config.OLLAMA_MODEL}`, or add ANTHROPIC_API_KEY to .env, then restart "
               "the app. Until then, questions show the retrieved passages only.")


@st.cache_resource
def get_retriever():
    return Retriever(mode="hybrid", rerank=True)


def render(result: dict) -> None:
    text = result["answer"]
    if result.get("search_query") and result["search_query"].lower() != result["question"].lower():
        st.caption(f"Understood as: {result['search_query']}")
    if not result["refused"]:
        for i, c in enumerate(result["sources"], 1):
            text = text.replace(f"[{i}]", f"[\\[{i}\\]]({c['url']})")
    st.markdown(text)
    if result.get("trimmed"):
        st.caption("Some sentences were removed because they had no supporting citation.")
    if result["pii_masked"]:
        st.warning("Personal identifiers in your message were removed before processing.")
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
    with st.expander("Closest pages" if result["refused"] else "Sources"):
        for i, c in enumerate(result["sources"], 1):
            if result["refused"] or i in cited:
                page = f", page {c['page']}" if c.get("page") else ""
                year = f", edition {c['edition_year']}" if c.get("edition_year") else ""
                st.markdown(f"**[{i}]** [{c['title'] or c['url']}]({c['url']}){year}{page}")


if "history" not in st.session_state:
    st.session_state.history = []

for role, payload in st.session_state.history:
    with st.chat_message(role):
        render(payload) if role == "assistant" else st.markdown(payload)

if question := st.chat_input("Ask about cricket rules, e.g. how many reviews does a team get in a Test?"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Searching ICC documents..."):
        if HAS_KEY:
            result = answer(question, get_retriever())
            render(result)
            st.session_state.history += [("user", result["question"]), ("assistant", result)]
        else:  # no LLM available: show what retrieval found, so the pipeline can still be checked
            st.markdown("**Search-only mode.** These are the passages the chatbot would answer from:")
            for i, c in enumerate(get_retriever().search(question), 1):
                page = f", page {c['page']}" if c.get("page") else ""
                page += f", edition {c['edition_year']}" if c.get("edition_year") else ""
                st.markdown(f"**[{i}]** [{c['title'] or c['url']}]({c['url']}){page} "
                            f"(relevance {c['score']:.2f})")
                st.caption(c["text"][:600])

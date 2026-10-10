from eval.run import mrr, recall_at_k, split
from cricketrag import config
from cricketrag.chunk.chunker import chunk_document, n_tokens
from cricketrag.crawl_scope import doc_id_for, in_scope
from cricketrag.extract.extract import extract_html, table_to_markdown
from cricketrag.generate.answer import REFUSAL_TEXT, answer, check_citations, mask_pii
from cricketrag.catalog import detect_format, detect_gender, detect_year, exclusion_reason
from cricketrag.retrieve.hybrid import edition_bonus, rrf

HTML = """<html><head><title>Topic 551</title></head><body><nav>skip me</nav><main>
<h1>Standard Deduction</h1><p>The standard deduction reduces taxable income. It varies by filing status.</p>
<h2>Amounts</h2><table><caption>2025 amounts</caption>
<tr><th>Status</th><th>Amount</th></tr><tr><td>Single</td><td>$15,000</td></tr></table>
<h2>Who cannot use it</h2><p>Some taxpayers must itemize. A nonresident alien is one example.</p>
</main></body></html>"""


def make_doc():
    title, blocks = extract_html(HTML)
    return {"doc_id": "a3f9c2e1", "url": "https://www.icc-cricket.com/about/cricket/rules-and-regulations/playing-conditions", "title": title,
            "content_hash": "h1", "blocks": blocks}


def test_extract_html_keeps_tables_and_drops_nav():
    doc = make_doc()
    assert doc["title"] == "Topic 551"
    assert [b["type"] for b in doc["blocks"]].count("table") == 1
    assert "skip me" not in str(doc["blocks"])
    assert "| Single | $15,000 |" in str(doc["blocks"])


def test_table_to_markdown_pads_ragged_rows():
    md = table_to_markdown([["a", "b"], ["1"]])
    assert md.splitlines()[2] == "| 1 |  |"


def test_struct_chunks_isolate_table_with_heading_path():
    chunks = chunk_document(make_doc(), "struct")
    tables = [c for c in chunks if c["content_type"] == "table"]
    assert len(tables) == 1
    assert tables[0]["text"].startswith("Topic 551\nStandard Deduction > Amounts\n2025 amounts")
    assert all(n_tokens(c["text"]) <= config.STRUCT_MAX_TOKENS for c in chunks)


def test_chunk_ids_are_stable_and_versioned():
    a, b = chunk_document(make_doc(), "struct"), chunk_document(make_doc(), "struct")
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
    assert a[0]["chunk_id"] == "a3f9c2e1:struct-v2:0000"
    assert chunk_document(make_doc(), "fixed")[0]["chunk_id"] == "a3f9c2e1:fixed-v1:0000"


def test_fixed_chunks_respect_size_and_overlap():
    doc = {"doc_id": "d", "url": "u", "content_hash": "h",
           "blocks": [{"type": "text", "text": " ".join(f"Sentence number {i} is here." for i in range(400))}]}
    chunks = chunk_document(doc, "fixed")
    assert len(chunks) > 3
    assert all(n_tokens(c["text"]) <= config.FIXED_TOKENS for c in chunks)
    assert chunks[0]["text"].split(". ")[-1] in chunks[1]["text"]  # overlap carried


def test_scope_and_doc_id():
    site = "https://www.icc-cricket.com"
    assert in_scope(site + "/about/cricket/rules-and-regulations/decision-review-system")
    assert in_scope("https://images.icc-cricket.com/image/upload/prd/a8114zro5tkioifz0jqg.pdf")
    assert in_scope(site + "/news/changes-to-playing-conditions-explained")
    assert in_scope(site + "/news/new-icc-playing-conditions-bring-key-changes-across-formats")
    assert not in_scope(site + "/news/india-beat-australia-by-five-wickets")
    assert not in_scope(site + "/news/pacer-ruled-out-as-india-shuffle-white-ball-squads")
    assert not in_scope(site + "/news/usa-and-uae-fined-for-slow-over-rates")
    assert not in_scope(site + "/tournaments/t20cricketworldcup")
    assert not in_scope(site + "/about/x?page=2")
    assert not in_scope("https://example.com/about/x.pdf")
    assert doc_id_for(site + "/a/") == doc_id_for(site + "/a#top")


def test_rrf_rewards_agreement():
    assert rrf([["a", "b", "c"], ["b", "d"]])[0] == "b"


def test_citation_check():
    assert check_citations("Single filers get $15,000 this year [1]. It varies by status [2][3].", 3) == []
    assert check_citations("Single filers get $15,000 this year.", 3)
    assert check_citations("Single filers get $15,000 this year [7].", 3)


def test_mask_pii():
    text, found = mask_pii("My SSN is 123-45-6789, email a@b.com")
    assert found and "123-45-6789" not in text and "[EMAIL]" in text


class FakeRetriever:
    def __init__(self, score):
        self.score = score

    def search(self, q):
        return [{"chunk_id": "d:struct-v1:0000", "url": "https://www.icc-cricket.com/x", "title": "T",
                 "text": "Single filers get $15,000.", "score": self.score}]


def test_answer_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_FILE", tmp_path / "log.jsonl")
    ok = answer("What is it?", FakeRetriever(0.9), llm=lambda s, u: "Single filers get $15,000 this year [1].")
    assert not ok["refused"]
    low = answer("What is it?", FakeRetriever(0.01), llm=lambda s, u: 1 / 0)  # LLM never called
    assert low["refused"] and low["reason"] == "low_retrieval_score" and low["answer"] == REFUSAL_TEXT
    tok = answer("What is it?", FakeRetriever(0.9), llm=lambda s, u: config.REFUSAL_TOKEN)
    assert tok["reason"] == "model_insufficient_context"
    bad = answer("What is it?", FakeRetriever(0.9), llm=lambda s, u: "Single filers get $15,000 this year.")
    assert bad["reason"] == "citation_check_failed"
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 4
    part = "Single filers get $15,000 this year [1]. Not covered: the amount for joint filers."
    assert check_citations(part, 1) == []


def test_rewrite_query_falls_back_safely():
    from cricketrag.generate.answer import rewrite_query
    assert rewrite_query("how many overs in ode?", lambda s, u: "How many overs are in an ODI?") == "How many overs are in an ODI?"
    assert rewrite_query("no ball?", lambda s, u: 1 / 0) == "no ball?"
    assert rewrite_query("no ball?", lambda s, u: "") == "no ball?"


def test_split_is_stratified_and_group_disjoint():
    items = [{"id": f"q{i}", "category": c, "gold_urls": [f"u{i // 2}"]}
             for c in ("fact", "table") for i in range(20)]
    s = split(items)
    assert len(s["dev"]) == 24 and len(s["test"]) == 16
    dev_urls = {(i["category"], i["gold_urls"][0]) for i in s["dev"]}
    assert not dev_urls & {(i["category"], i["gold_urls"][0]) for i in s["test"]}
    assert recall_at_k(["a", "b"], ["b"]) == 1.0 and mrr(["a", "b"], ["b"]) == 0.5


def test_catalog_detection():
    t = "ICC Women's Test Match playing Conditions - October 2026"
    assert (detect_year(t, ""), detect_format(t), detect_gender(t)) == (2026, "Test", "women")
    assert detect_year("Code of Conduct", "Effective as from 16 June 2023. Amended 2019 and 2019.") == 2023
    assert detect_format("ICC Men's T20 World Cup 2024 Playing Conditions") == "T20I"
    assert exclusion_reason("Decision in the matter of A Player - 12 July 2024", "", "")
    assert exclusion_reason("ICC Annual Report 2011-12", "/about/the-icc/publications/annual-report", "")
    assert not exclusion_reason("ICC Men's ODI Playing Conditions", "/about/cricket/rules-and-regulations", "")


def test_edition_bonus_prefers_newest_or_named_year():
    assert edition_bonus(2026, set()) > edition_bonus(2013, set()) > 0
    assert edition_bonus(2013, {2013}) > edition_bonus(2026, {2013}) == 0
    assert edition_bonus(None, set()) == 0


def test_superseded_editions():
    from cricketrag.catalog import family, mark_superseded
    assert family("ICC Men\u2019s ODI Playing Conditions - October 2026") == family("ICC Men's ODI Playing Conditions 2023")
    cat = {k: {"title": t, "edition_year": y, "excluded": ""} for k, (t, y) in {
        "a": ("ICC Men's ODI Playing Conditions - October 2026", 2026),
        "b": ("ICC Men's ODI Playing Conditions - 2023", 2023),
        "c": ("ICC Playing Handbook 2019-20", 2019),
        "d": ("Anti-Discrimination Code (effective 1 August 2019)", 2019),
        "e": ("Formats of the Game", None)}.items()}
    mark_superseded(cat)
    assert [cat[k]["superseded"] for k in "abcde"] == [False, True, True, False, False]


def test_citation_check_tolerates_common_layouts():
    ok = [
        "A no ball is called for overstepping. [1] The batting side gets one run. [2]",
        "A no ball is called in several cases:\n- The bowler oversteps the popping crease [1].\n- The ball bounces more than once before reaching the striker [2].",
        "Fielding limits apply, e.g. Only two fielders may stand behind square on the leg side [3].",
        "**No ball**\nThe umpire calls and signals no ball when the bowler oversteps [1].",
    ]
    for text in ok:
        assert check_citations(text, 3) == [], text
    assert check_citations("A no ball is called for overstepping [1]. The batting side gets one run.", 3)
    assert check_citations("- The bowler oversteps the popping crease.\n- The ball bounces twice [2].", 3)


def test_citation_repair_and_trim(tmp_path, monkeypatch):
    from cricketrag.generate.answer import keep_cited
    monkeypatch.setattr(config, "LOG_FILE", tmp_path / "log.jsonl")
    draft = "A no ball is an unfair delivery. The umpire signals it [1]. Not covered: every mode of no ball."
    assert keep_cited(draft, 1) == "The umpire signals it [1]. Not covered: every mode of no ball."
    assert keep_cited("Nothing here is cited at all.", 1) == ""
    replies = iter(["Understood question?", draft, config.REFUSAL_TOKEN])   # rewrite, draft, failed repair
    res = answer("no bal", FakeRetriever(0.9), llm=lambda s, u: next(replies))
    assert not res["refused"] and res["trimmed"] and res["answer"].startswith("The umpire signals it [1].")
    replies = iter(["Understood question?", draft, "An unfair delivery is a no ball [1]. The umpire signals it [1]."])
    res = answer("no bal", FakeRetriever(0.9), llm=lambda s, u: next(replies))
    assert res["answer"].startswith("An unfair delivery") and "trimmed" not in res


def test_passage_key_ignores_title():
    from cricketrag.retrieve.hybrid import passage_key
    a = {"title": "Men's ODI Playing Conditions", "heading_path": "Page 40", "text": "Men's ODI Playing Conditions\nPage 40\n21.1 Mode of delivery The umpire shall ascertain"}
    b = {"title": "Women's T20I Playing Conditions", "heading_path": "Page 38", "text": "Women's T20I Playing Conditions\nPage 38\n21.1 Mode of delivery The umpire shall ascertain"}
    assert passage_key(a) == passage_key(b)


def test_plan_query_and_multi_search(tmp_path, monkeypatch):
    from cricketrag.generate.answer import plan_query
    reply = "How do powerplay rules differ between ODI and T20I?\nSEARCH: ODI powerplay overs\nSEARCH: T20I powerplay overs"
    q, subs = plan_query("powerplay odi vs t20", lambda s, u: reply)
    assert q.startswith("How do powerplay") and subs == ["ODI powerplay overs", "T20I powerplay overs"]
    assert plan_query("no ball?", lambda s, u: "What is a no ball?") == ("What is a no ball?", [])
    assert plan_query("no ball?", lambda s, u: "SEARCH: no ball") == ("no ball?", [])

    from cricketrag.retrieve.hybrid import Retriever
    class Stub(Retriever):
        def __init__(self): pass
        def search(self, query, top_k=5):
            tag = "odi" if "ODI" in query and "T20I" not in query else "t20" if "T20I" in query and "ODI" not in query else "both"
            return [{"chunk_id": f"{tag}{i}", "title": "", "heading_path": "", "text": f"{tag} passage {'abcdefgh'[i]} text",
                     "score": 0.9 - i / 10} for i in range(top_k)]
    got = [c["chunk_id"] for c in Stub().search_multi("ODI and T20I powerplay", ["ODI powerplay", "T20I powerplay"])]
    assert len(got) == 6 and sum(g.startswith("odi") for g in got) == 2 and sum(g.startswith("t20") for g in got) == 2

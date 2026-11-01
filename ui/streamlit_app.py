"""Streamlit UI.  Run:  streamlit run ui/streamlit_app.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# Streamlit Cloud / HF Spaces secrets -> env vars (config.py env se padhta hai)
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:  # noqa: BLE001 - secrets file na ho to chalega
    pass

import pandas as pd  # noqa: E402

from app.extraction import ExtractionError  # noqa: E402
from app.monitoring import summarize  # noqa: E402
from app.resume_match import extract_resume_text  # noqa: E402
from app.service import get_service  # noqa: E402

st.set_page_config(page_title="Job Post Intelligence Engine", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def svc():
    return get_service()


page = st.sidebar.radio("Page", ["➕ Add posts", "📋 Jobs", "🔎 Search", "📄 Resume match", "🗄️ Ask SQL", "📈 Monitoring"])
st.sidebar.caption(f"LLM: {svc().llm.provider}:{svc().llm.model}")

if page == "➕ Add posts":
    st.header("Raw post paste karo")
    st.caption("Ek se zyada posts ho to unke beech me ek line `---` daalo.")
    text = st.text_area("Job post(s)", height=260)
    if st.button("Extract & save", type="primary") and text.strip():
        for chunk in [c.strip() for c in text.split("\n---\n") if c.strip()]:
            try:
                with st.spinner("Extracting..."):
                    r = svc().ingest(chunk)
                label = {"new": "✅ New job saved", "duplicate": f"♻️ Duplicate of #{r.get('duplicate_of')}",
                         "exact_duplicate": f"♻️ Exact repost of #{r.get('duplicate_of')}"}[r["status"]]
                st.success(f"{label} (post #{r['post_id']}, attempts: {r.get('attempts', '-')})")
                st.json(r["job"], expanded=False)
            except ExtractionError as e:
                st.error(f"Validation failed after retries: {e}")
                st.write(e.errors)
            except Exception as e:  # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")

elif page == "📋 Jobs":
    st.header("Saved jobs")
    show_dups = st.checkbox("Show duplicate reposts too", value=False)
    posts = svc().db.list_posts(limit=1000, include_duplicates=show_dups)
    st.write(f"{len(posts)} posts · {svc().vstore.count()} vectors")
    if posts:
        df = pd.DataFrame(posts).drop(columns=["raw_text", "content_hash"])
        df["skills"] = df["skills"].apply(", ".join)
        df["batch"] = df["batch"].apply(lambda b: ",".join(map(str, b)))
        st.dataframe(df, use_container_width=True, hide_index=True)
    if st.button("Re-cluster all posts (dedup)"):
        st.json(svc().recluster())

elif page == "🔎 Search":
    st.header("Sawaal puchho (Hinglish chalega)")
    q = st.text_input("Query", placeholder="pune me 25k se upar AI internship")
    k = st.slider("Results", 1, 10, 5)
    if st.button("Search", type="primary") and q:
        out = svc().search(q, k)
        st.markdown(out["answer"])
        with st.expander("Understood as"):
            st.json(out["parsed"])
            if out["filters_relaxed"]:
                st.warning("Filters bahut tight the, isliye relax karke search kiya.")
        st.subheader("Sources")
        for s in out["sources"]:
            st.markdown(f"**[{s['id']}] {s['role']} @ {s['company']}** — {s['location'] or 'Remote'} · "
                        f"{s['stipend_min'] or '?'}–{s['stipend_max'] or '?'}/mo · sim {s['similarity']}  \n"
                        f"{s['apply_link'] or s['apply_email'] or ''}")

elif page == "📄 Resume match":
    st.header("Resume vs jobs")
    up = st.file_uploader("Resume (PDF/TXT)", type=["pdf", "txt", "md"])
    if up and st.button("Match", type="primary"):
        text = extract_resume_text(up.name, up.getvalue())
        res = svc().match_resume(text) if text.strip() else []
        if not res:
            st.warning("Resume se text nahi mila, ya DB me jobs nahi hain.")
        else:
            df = pd.DataFrame(res)
            df["matched_skills"] = df["matched_skills"].apply(", ".join)
            df["missing_skills"] = df["missing_skills"].apply(", ".join)
            st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "🗄️ Ask SQL":
    st.header("Plain-English → SQL")
    st.caption("Sirf SELECT, sirf `jobs` view. Error aaye to 3 retries.")
    q = st.text_input("Question", placeholder="How many internships are in Pune?")
    if st.button("Run", type="primary") and q:
        r = svc().ask_sql(q)
        if r["success"]:
            st.code(r["sql"], language="sql")
            st.dataframe(pd.DataFrame(r["rows"], columns=r["columns"]), use_container_width=True)
            st.caption(f"attempts: {r['attempts']}")
        else:
            st.error("SQL generate nahi ho paayi.")
        if r["errors"]:
            with st.expander("Retry log"):
                st.write(r["errors"])

elif page == "📈 Monitoring":
    st.header("LLM monitoring")
    logs = svc().db.fetch_logs()
    s = summarize(logs)
    if not s["total_calls"]:
        st.info("Abhi koi LLM call log nahi hui.")
    else:
        c = st.columns(5)
        c[0].metric("Calls", s["total_calls"])
        c[1].metric("Error rate", f"{s['error_rate'] * 100:.1f}%")
        c[2].metric("Avg latency", f"{s['avg_latency_ms']:.0f} ms")
        c[3].metric("p95 latency", f"{s['p95_latency_ms']:.0f} ms")
        c[4].metric("Tokens", s["prompt_tokens"] + s["completion_tokens"])
        st.subheader("Per day")
        st.bar_chart(pd.DataFrame(s["daily"]).set_index("date")[["calls", "tokens"]])
        for title, key in [("By purpose", "by_purpose"), ("By model", "by_model"), ("By prompt version", "by_prompt_version")]:
            st.subheader(title)
            st.dataframe(pd.DataFrame(s[key]), use_container_width=True, hide_index=True)
        if s["recent_errors"]:
            st.subheader("Recent errors")
            st.dataframe(pd.DataFrame(s["recent_errors"]), use_container_width=True, hide_index=True)
        with st.expander("Raw call log (prompt/response)"):
            st.dataframe(pd.DataFrame(logs[:100]), use_container_width=True)

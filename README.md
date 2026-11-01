# 🧠 Job Post Intelligence Engine

Telegram/WhatsApp groups ki messy (Hinglish, emoji-bhari) job posts ko LLM se **validated structured data** me badalta hai, aur uske upar **dedup, RAG search, resume match aur text-to-SQL** deta hai. Har LLM call monitor hoti hai.

```
Raw post ─► LLM extraction ─► Pydantic validation (fail ⇒ feedback ke saath retry) ─► SQLite
                                                        └─► embedding ─► Chroma ─► dedup cluster
Query (Hinglish) ─► LLM query-parse (filters) ─► Chroma filtered search ─► grounded answer + sources
Question ─► LLM ─► SQL ─► SELECT-only guard ─► read-only SQLite (error ⇒ 3 retries)
Har LLM call ─► llm_logs table ─► Monitoring page
```

## Features → code

| Feature | Kahan hai |
|---|---|
| Extraction + retry | `app/extraction.py`, prompts: `prompts/extract_v1..v3.txt` |
| Validation | `app/schemas.py` (`JobPost`: stipend int, valid URL, batch 2015–2035, min ≤ max) |
| Database | `app/db.py` (tables `posts`, `llm_logs`, view `jobs`) |
| Dedup | `app/dedup.py` + `app/service.py` (cosine ≥ 0.88 **and** same company) |
| RAG search | `app/search.py` (Hinglish → English query + filters → Chroma) |
| Resume match | `app/resume_match.py`, `app/matching.py` (0.6 × embedding + 0.4 × keyword overlap) |
| Text-to-SQL | `app/text2sql.py`, `app/sqlguard.py` (3 retries, SELECT-only) |
| Monitoring | `app/llm.py` (logging), `app/monitoring.py`, Streamlit "Monitoring" page |
| API / UI | `app/api.py` (FastAPI), `ui/streamlit_app.py` |
| Eval | `eval/` |

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # GROQ_API_KEY ya GEMINI_API_KEY daalo (dono ka free tier hai)
streamlit run ui/streamlit_app.py      # UI
uvicorn app.api:app --reload           # (optional) API -> http://localhost:8000/docs
pytest -q                              # tests (koi API key / model download nahi chahiye)
```

Pehli baar real embeddings chalane par `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) download hota hai.
Sirf try karna ho to `.env` me `EMBED_BACKEND=hash` rakho (offline, low quality).

**Local open-source model (Ollama):** `ollama pull qwen2.5:3b`, phir `.env`: `LLM_PROVIDER=ollama`.

> Model names (Groq/Gemini) kabhi-kabhi deprecate hote hain. Agar 404/model-not-found aaye to `.env` me `GROQ_MODEL` / `GEMINI_MODEL` provider ki current model list se update karo.

## Safety notes

- **Text-to-SQL ke 4 layers**: (1) `sqlguard` sirf single SELECT/WITH allow karta hai, comments/`;`/DML/PRAGMA block; (2) sirf `jobs` view readable (raw `posts`/`llm_logs` blocked); (3) DB read-only connection (`mode=ro` + `query_only`); (4) SQLite authorizer sirf SELECT/READ allow karta hai + 5s timeout + auto `LIMIT 200`.
- API keys sirf environment/secrets me. `.env` `.gitignore` me hai.
- Posts me personal data (phone/email) ho sakta hai — public deploy karne se pehle sochlo.

## Evaluation (project ko standout banane wala hissa)

Repo me **starter dataset** hai: `eval/labelled_posts.json` (50 posts, 6 reposts ke saath) — ye `eval/make_synthetic_dataset.py` se banta hai, **synthetic** hai.
Resume/interview me credible claim ke liye **apni real Telegram posts se kam se kam 25–30 posts khud label karke** isi format me file me daalo (`text` + `labels`, jaise existing entries).

```bash
# 1) Prompt versions compare (ek model, 3 prompts)
python -m eval.run_extraction_eval --models groq:llama-3.3-70b-versatile --prompts extract_v1 extract_v2 extract_v3

# 2) Best prompt ke saath open-source vs API model (accuracy / latency / cost)
python -m eval.run_extraction_eval --prompts extract_v2 \
    --models groq:llama-3.3-70b-versatile ollama:qwen2.5:3b gemini:gemini-2.5-flash

# 3) Text-to-SQL: 40 questions, execution accuracy
python -m eval.run_text2sql_eval --provider groq
```

Output: `eval/results/latest_extraction.md` (per-field accuracy table), `eval/results/text2sql_latest.json`.
`--sleep 2` free-tier rate limits ke liye hai; zyada 429 aaye to badha do.
Cost column list-price estimate hai (`app/config.py → PRICING`); free tier me actual cost 0 hota hai.

**Resume me kya likhna hai (numbers apne run ke daalna):**
> Built an LLM pipeline that converts unstructured Hinglish job posts into Pydantic-validated JSON (**XX% field accuracy** on N hand-labelled posts; best of 3 prompt versions: `extract_vX`); embedding-based dedup, hybrid RAG search, resume matching and a guarded text-to-SQL interface (**YY/40** execution accuracy); every LLM call logged for latency/token/error monitoring. Compared Llama-3.3-70B (Groq) vs Qwen2.5-3B (local) on accuracy, latency and cost.

## Deployment

- **MVP (free):** Streamlit Community Cloud ya Hugging Face Spaces (Streamlit/Docker SDK) — main file `ui/streamlit_app.py`. UI seedha `app.service` use karta hai, to alag backend zaruri nahi.
  Secrets: `GROQ_API_KEY`, `GEMINI_API_KEY`, `LLM_PROVIDER` platform ke **Secrets** me daalo.
  Memory kam ho (torch heavy) to `EMBED_BACKEND=hash` ya chhota embedding model use karo.
- **Backend alag:** `docker build -t jobintel . && docker run -p 8000:8000 --env-file .env jobintel` → Render (Docker) par deploy. Free tier idle par so jata hai (pehla request slow). **Note:** Render/Spaces ka free disk ephemeral hota hai — SQLite/Chroma restart par reset ho sakte hain; isliye Postgres (neeche) ya persistent disk use karo.
- **CI:** `.github/workflows/tests.yml` har push par pytest chalata hai (hash embedder + in-memory vector store + fake LLM, to keys nahi chahiye).

## Roadmap: SQLite → PostgreSQL (Supabase)

Abhi `app/db.py` stdlib `sqlite3` use karta hai (+ SQLite-specific `json_each`). Migration steps:
1. `psycopg`/SQLAlchemy se `Database` class ka Postgres version (same method names).
2. `jobs` view me `json_each` → `jsonb_array_elements_text` + `string_agg`.
3. Text-to-SQL ka read-only user/role (`GRANT SELECT ON jobs`) — sqlite authorizer ki jagah.
4. `DATABASE_URL` env var.

Aur ideas: dedup eval (pairwise precision/recall), Telegram bot ingestion (`python-telegram-bot`/Telethon), prompt A/B runtime pe.

## Project layout

```
app/        core logic (+ api.py FastAPI)      prompts/   versioned prompt files
ui/         Streamlit app                      eval/      datasets + eval scripts
tests/      pytest                              data/      sqlite + chroma (gitignored)
```

## Known limitations

- Hash embedder sirf dev/test ke liye hai; real similarity ke liye multilingual sentence-transformer use karo. `DEDUP_THRESHOLD` (0.88) ko apne data par tune karo.
- Resume match score ek heuristic hai (weights 0.6/0.4 aur cosine rescale range `app/matching.py` me tunable).
- Pehli baar jab ye code real API keys ke saath chalega to model names / rate limits adjust karne pad sakte hain.

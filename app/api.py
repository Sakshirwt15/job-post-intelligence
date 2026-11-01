"""FastAPI backend.  Run:  uvicorn app.api:app --reload"""
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.extraction import ExtractionError
from app.monitoring import summarize
from app.resume_match import extract_resume_text
from app.schemas import BulkIngestRequest, IngestRequest, SearchRequest, SQLRequest
from app.service import get_service

app = FastAPI(title="Job Post Intelligence Engine", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/posts")
def ingest(req: IngestRequest):
    try:
        return get_service().ingest(req.raw_text, req.prompt_version)
    except ExtractionError as e:
        raise HTTPException(422, {"message": str(e), "errors": e.errors})
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/posts/bulk")
def ingest_bulk(req: BulkIngestRequest):
    return get_service().ingest_many(req.posts)


@app.get("/posts")
def list_posts(limit: int = 100, offset: int = 0, include_duplicates: bool = False):
    return get_service().db.list_posts(limit, offset, include_duplicates)


@app.get("/posts/{post_id}")
def get_post(post_id: int):
    post = get_service().db.get_post(post_id)
    if not post:
        raise HTTPException(404, "Post not found")
    return post


@app.post("/dedup/recluster")
def recluster():
    return get_service().recluster()


@app.post("/search")
def search(req: SearchRequest):
    return get_service().search(req.query, req.k)


@app.post("/sql")
def sql(req: SQLRequest):
    return get_service().ask_sql(req.question)


@app.post("/match")
def match(resume: UploadFile = File(...), top_n: int = 20):
    text = extract_resume_text(resume.filename or "resume.txt", resume.file.read())
    if not text.strip():
        raise HTTPException(400, "Could not read any text from the resume")
    return get_service().match_resume(text, top_n)


@app.get("/monitoring")
def monitoring():
    return summarize(get_service().db.fetch_logs())

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
# CPU-only torch chhota hota hai (sentence-transformers ke liye)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
COPY . .
EXPOSE 8000 8501
# Default: API. UI ke liye: docker run ... streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

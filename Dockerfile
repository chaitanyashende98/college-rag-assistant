
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY embeddings.py .
COPY vector_store.py .
COPY retriever.py .
COPY rag_service.py .
COPY mcp_server.py .

EXPOSE 8080

CMD ["python", "mcp_server.py"]

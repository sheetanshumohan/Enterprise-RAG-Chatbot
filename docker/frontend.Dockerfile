FROM python:3.11-slim

WORKDIR /app

COPY apps/frontend/pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit httpx httpx-sse

COPY apps/frontend/app.py ./
COPY apps/frontend/api_client.py ./
COPY apps/frontend/pages ./pages
COPY apps/frontend/.streamlit ./.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

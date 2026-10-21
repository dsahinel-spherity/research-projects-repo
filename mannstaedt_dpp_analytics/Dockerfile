FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Cache dependencies separately from application changes.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY app/ ./app/
COPY analytics_engine/ ./analytics_engine/
COPY data/mock/ ./data/mock/
COPY .streamlit/config.toml ./.streamlit/config.toml
RUN python -m pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

USER appuser
EXPOSE 8501

# Checks Streamlit availability, not Umami credentials or upstream availability.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"

CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

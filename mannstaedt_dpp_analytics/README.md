# Mannstaedt DPP Analytics

A Streamlit dashboard for GMH Digital Product Passport usage, backed by real
Umami analytics. Business pages show section engagement, document opens, QR/web
journeys and behavior signals. The Analytics Lab and Dataset Explorer use
separate synthetic datasets; their model results are experimental.

## Run locally

From this project directory, with an existing configured `.env`:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app/dashboard.py
```

Open **http://localhost:8501**. For first-time setup, see the
[user manual](docs/USER_MANUAL.md). Never commit `.env` or credentials.

## Docker Compose

With Docker running and `.env` configured:

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 mannstaedt-analytics
```

Open **http://localhost:8501**. Stop with `docker compose down`.
The single service runs Streamlit as a non-root user and reads credentials at
runtime. No database, FastAPI service or source-code volume is required.

## Docker without Compose

```bash
docker build -t mannstaedt-dpp-analytics .
docker run -d --name mannstaedt-analytics --env-file .env \
  -p 8501:8501 --restart unless-stopped mannstaedt-dpp-analytics
```

The [user manual](docs/USER_MANUAL.md) covers configuration, dashboard pages,
restarts, rebuilds, troubleshooting and developer notes.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

Tests run offline with synthetic fixtures. Real GMH data is never used to train
the synthetic Lab models. The older traffic forecast remains in `ml_service.py`
but is not connected to the current dashboard.

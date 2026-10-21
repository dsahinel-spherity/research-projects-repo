# Mannstaedt DPP Analytics — user manual

## What the app does

This dashboard helps the GMH team understand how visitors use Digital Product
Passports (DPPs): which sections they view, how they arrive, which documents they
open, and how they navigate. Real business analytics come from nonprod Umami.
The separate Analytics Lab demonstrates machine learning with synthetic data.

## Prerequisites

- Access to this repository and the configured Umami service.
- For containers: Docker Engine or Docker Desktop running, with Docker Compose v2
  or newer (`docker compose version`).
- For local Python development: Python 3.9 or newer and pip. The container uses
  Python 3.12. Keep the existing `.venv`; Docker does not use or change it.
- Port 8501 available. Do not run local Python, standalone Docker and Compose
  simultaneously on the same host port.

Run the commands below from:

```bash
cd ~/Desktop/research-projects-repo/mannstaedt_dpp_analytics
```

## Configure the environment

Keep your existing `.env` file unchanged. If setting up a new checkout without
one, copy `.env.example` to `.env`, then fill in the values locally:

```bash
# Only on a new checkout without .env:
cp -n .env.example .env
```

| Variable | Purpose |
| --- | --- |
| `UMAMI_BASE_URL` | Base URL of the nonprod Umami service, without a trailing slash |
| `UMAMI_WEBSITE_ID` | Website whose events should be queried |
| `UMAMI_USERNAME` | Authorized Umami login username |
| `UMAMI_PASSWORD` | Authorized Umami login password |
| `DPP_VIEWER_BASE_URL` | Optional public viewer origin for DPP links; defaults to `https://dpp.vera.spherity.dev` |

Replace the example viewer URL or omit that optional line to use the default.
There is no need to supply `TOKEN`: the service logs in and reuses its token.

Local Python reads `.env` through `python-dotenv`. Existing process environment
variables take precedence. Compose and `docker run --env-file` inject values at
runtime; `.env` is not copied into the image. Use simple `KEY=value` lines
compatible with your chosen launcher. For passwords containing dollar signs,
Compose supports single-quoted literal values; `docker run --env-file` has
different quote handling, so validate the chosen mode without printing secrets.

**Never commit credentials.** `.env` and `.env.*` are ignored by Git, except
placeholder examples/templates. Docker excludes all environment files and
Streamlit secrets. Do not paste resolved environment output into tickets or logs.
Use `docker compose config --quiet` to validate without printing credentials.

Umami timeout (30 seconds), retry attempts (4), real-data cache (300 seconds) and
Lab experiment cache (600 seconds) are currently code constants, not environment
settings. Temporary 429/500/502/503/504 responses are retried. Token refresh is
handled after a 401 response.

## Run locally with Python

For an existing development environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app/dashboard.py
```

For a fresh checkout only, create the environment first with
`python3 -m venv .venv`. Do not replace an existing environment unnecessarily.
Use `python -m streamlit` from the project root to avoid `No module named app`.

Open **http://localhost:8501**. Stop with Ctrl+C. Restart by repeating the last
command. Refresh the browser after code changes.

## Run with Docker Compose

```bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

Open **http://localhost:8501**. The service is `mannstaedt-analytics`.
The container binds to `0.0.0.0:8501` internally and maps host port 8501.
It runs as an unprivileged user, with no source mount or persistent volume.
Synthetic CSVs are included in the image; Umami data is fetched at runtime.

Common commands:

```bash
# Follow application logs (Ctrl+C exits the log viewer, not the app)
docker compose logs -f --tail=100 mannstaedt-analytics

# Restart the same container
docker compose restart mannstaedt-analytics

# Stop / start without removing the container
docker compose stop
docker compose start

# Stop and remove this project's containers/network
docker compose down

# Rebuild and recreate after source or dependency changes
docker compose up -d --build

# Recreate after changing runtime environment values
docker compose up -d --force-recreate
```

`restart: unless-stopped` restarts the service after failures or a Docker restart,
unless you explicitly stopped it. An unhealthy status alone does not trigger a
Docker restart.

## Run with Docker without Compose

```bash
docker build -t mannstaedt-dpp-analytics .
docker run -d --name mannstaedt-analytics \
  --env-file .env -p 8501:8501 --restart unless-stopped \
  mannstaedt-dpp-analytics

docker logs --tail=100 mannstaedt-analytics
docker stop mannstaedt-analytics
docker start mannstaedt-analytics
```

After a code change, rebuild the image, then replace the old container:

```bash
docker build -t mannstaedt-dpp-analytics .
docker stop mannstaedt-analytics
docker rm mannstaedt-analytics
# Repeat the docker run command above.
```

## Check startup and health

```bash
curl --fail http://localhost:8501/_stcore/health
```

A successful response indicates Streamlit is serving requests. It does **not**
verify Umami credentials, correct event metadata or successful page rendering.
Open Overview to check real-data access, and Analytics Lab to check synthetic
experiments. Docker also runs this health check every 30 seconds, after a startup
grace period. Inspect health with `docker compose ps`.

The supplied port mapping exposes the app on the host's network interfaces.
For a machine restricted to local access, use `127.0.0.1:8501:8501` instead.
Before exposing a server publicly, provide access control and HTTPS through your
existing deployment infrastructure. This app does not add a login gateway.

## Use the dashboard

Select the real-data date range in the sidebar. Dates are converted to UTC day
boundaries. Choose **All GMH DPPs** or one DPP, then use the navigation:

| Page | What to do |
| --- | --- |
| Overview | Review usage, QR/web comparisons, document-open rates and behavior signals |
| Engagement | Compare sections, documents, DPPs, devices and countries |
| User Journey & Traceability | Inspect ordered paths, entry/exit sections, transitions and individual journeys |
| Analytics Lab | Choose a synthetic question and press **Run synthetic experiment**; expand cleaning/filtering details before reviewing results |
| Dataset Explorer | Inspect synthetic data distributions and optional raw/prepared tables |

Ranked charts initially show ten categories. Use the show-all control for larger
lists; expand tables when you need records. Behavior-signal records include DPP
identifiers and links when the recorded path is usable.

### GMH scope and journeys

The organization is fixed to `gmh-gruppe-d034406b`. There is no general
organization selector. Both the all-DPP and single-DPP views use matching GMH
custom events and attributable pageviews; another DPP's document opens must not
be counted just because it shared a session. Missing or ambiguous attribution
can exclude pageviews.

Traceability here means visitor navigation: entry → sections → document opens →
last observed activity. It does not mean supplier-chain risk. Journeys split only
after more than ten minutes of inactivity, not because a DID is URL-encoded.
Duration measures the span between recorded events, not active reading time.
Journeys without tracked section/document interactions are excluded.

All-DPP journeys can touch multiple DPPs. Their displayed identifier/link refers
to the first recorded DPP; select an individual DPP for its own scoped activity.

**Document availability matters:** Mannstaedt DPPs currently have no documents.
Zero document opens are therefore expected. Tracking will count properly
attributed `dpp-document-open` events when documents are added. A behavior signal
such as “Exploration without a document” means at least three section visits and
no document open; it is a subset of all journeys without documents. These signals
are investigation prompts, not proof of a usability problem.

### Real analytics versus synthetic ML

The first three pages use real Umami events. The Lab and Explorer use separate
synthetic datasets and do not train on real GMH history. Real-data date/DPP filters
do not change synthetic datasets. They remain usable if Umami is unavailable.

Guided experiments cover document opening, deep engagement, journey duration,
section depth and unusual journeys. Supervised results include a held-out split,
a simple baseline and train-only preprocessing. The cleaning expander explains
removed duplicates, missing outcomes, selected/excluded inputs, replacements,
encoding and the split. Unusual-journey detection is an in-sample review ranking.
Advanced models remain available for exploratory use. Synthetic scores do not
validate predictions for real users or establish production model readiness.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Cannot connect to Docker daemon | Start Docker Desktop/Engine and check Docker access |
| Port 8501 already allocated | Stop the existing local app/container or change the host side of the mapping |
| Compose cannot find `.env` | Run from the project root and create a local file from the example if needed |
| `No module named app` locally | Run from the project root with `python -m streamlit`; install with `python -m pip install -e .` |
| Real analytics fails | Check runtime Umami settings, network/VPN access, website ID and account permissions; do not share credentials in logs |
| Empty business charts | Check date range, selected DPP and GMH custom-event metadata; empty is not a reason to substitute synthetic data |
| Old data appears | Use **Refresh real data** or wait five minutes; rebuild containers for code changes |
| Container is healthy but analytics fails | Health checks Streamlit, not Umami; inspect the business page and logs |
| Dependency installation fails | Inspect the build's package error; dependencies include pinned packages and some unpinned ML libraries |

## Developer notes

```text
app/dashboard.py                 Streamlit entry point and sidebar
app/dashboard_views.py           Chart-first views and guided Lab UI
app/config.py                    Environment configuration
app/services/umami_service.py    Auth, pagination, retries, pivot enrichment
app/services/data_service.py     Real/synthetic dataset loaders
app/services/dpp_filter_service.py  Event ownership and DPP filtering
app/services/analytics_service.py   Business metrics
app/services/journey_service.py     Journey reconstruction and links
app/services/data_preparation_service.py  Advanced synthetic preprocessing
app/services/lab_service.py      Guided synthetic experiments and cleaning audit
app/services/engine_service.py   Adapter to the reusable model engine
analytics_engine/               Algorithm implementations
data/mock/                      Synthetic CSVs
.streamlit/config.toml          GMH theme
tests/                          Offline regression tests
```

`app/main.py` and `app/routes/analytics.py` are a separate, currently stale FastAPI
path, not required by the Streamlit container. `ml_service.py` retains an older
traffic forecast that is not wired into the current dashboard.

Umami uses `dpp-section-view` and `dpp-document-open`. Custom metadata is decoded
from `event-data-pivot` and joined by event ID; avoid per-event HTTP calls.
Authentication tokens are reused. Real loads are cached in Streamlit for five
minutes and guided experiments for ten minutes. The refresh button clears
Streamlit data caches; restarting clears in-memory caches and session results.
Caches and trained experimental models are not persisted across containers.

Run checks in the local virtual environment:

```bash
python -B -m unittest discover -s tests -v
python -m compileall -q app analytics_engine tests
git diff --check
docker compose config --quiet
```

Tests use offline fixtures and do not require access to Umami. Tests and docs are
excluded from the runtime image. Container logs are available with
`docker compose logs --tail=100 mannstaedt-analytics`; review them for sensitive
content before sharing. Avoid `docker inspect` environment dumps and unredacted
`docker compose config` output.

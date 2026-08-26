# VERA DPP Analytics Dashboard

A lightweight analytics dashboard for Vera Digital Product Passports using data from the Umami analytics platform.

The application extracts Umami data, processes it in Python, and displays the results in a Streamlit dashboard. The complete application can be packaged and run inside Docker for a reproducible setup.

## Current Features

- Pageviews
- Visitors
- Visits
- Bounce rate
- Average visit duration
- Traffic over time
- Most viewed DPP pages
- Visitor countries
- Browser distribution
- Referrers
- Realtime activity
- Session lookup

## Architecture

```text
Umami API
   ↓
Python data extraction
   ↓
Analytics processing
   ↓
Streamlit dashboard
   ↓
Docker container
```

## Project Structure

```text
user_analysis/
├── app/
│   ├── assets/
│   │   └── vera_logo.png
│   ├── services/
│   │   ├── __init__.py
│   │   ├── analytics_service.py
│   │   └── umami_service.py
│   ├── __init__.py
│   └── dashboard.py
├── .dockerignore
├── .env
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```text
UMAMI_BASE_URL=https://stats.vera.spherity.dev
UMAMI_WEBSITE_ID=YOUR_WEBSITE_ID
UMAMI_USERNAME=YOUR_USERNAME
UMAMI_PASSWORD=YOUR_PASSWORD
```

Do not commit the `.env` file to Git.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The dashboard is normally available at:

```text
http://localhost:8501
```

## Docker

Build the image:

```bash
docker build -t vera-dpp-analytics .
```

Run the container:

```bash
docker run \
  --env-file .env \
  -p 8502:8501 \
  --name vera-dpp-dashboard \
  vera-dpp-analytics
```

Open:

```text
http://localhost:8502
```

Run in the background:

```bash
docker run -d \
  --env-file .env \
  -p 8502:8501 \
  --restart unless-stopped \
  --name vera-dpp-dashboard \
  vera-dpp-analytics
```

Stop:

```bash
docker stop vera-dpp-dashboard
```

Start again:

```bash
docker start vera-dpp-dashboard
```

Remove:

```bash
docker rm vera-dpp-dashboard
```

## Current Umami Data Used

- Overall website statistics
- Pageviews over time
- Page paths
- Countries
- Browsers
- Referrers
- Realtime activity
- Session information

## Future Extensions

Once additional DPP interaction events are implemented, the dashboard can be extended with:

- QR entry tracking
- DPP section engagement
- Document views
- PDF downloads
- QR sessions with vs. without PDF downloads
- User journey analysis
- Transition analysis between DPP sections and documents

## Notes

The local `.venv` folder is only used for development and is not copied into Docker.

The Docker image installs Python dependencies from `requirements.txt`.

The `.env` file is excluded from the Docker image and passed to the container at runtime.

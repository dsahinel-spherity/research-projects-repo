# Mannstaedt GmbH DPP Analytics Dashboard

A company-specific analytics dashboard for Mannstaedt GmbH built on top of the VERA Umami analytics data.

The application focuses on Digital Product Passport usage and extends the standard Umami view with company-specific QR analysis, audience insights, and an experimental machine-learning traffic forecast.

## Current Features

- Mannstaedt GmbH branding
- Company-specific DPP analytics
- QR vs non-QR traffic comparison
- QR traffic share
- Average visit duration for QR vs non-QR traffic
- Device type analytics
- Region analytics
- DPP traffic over time
- DPP performance comparison
- Experimental 7-day ML traffic forecast
- Visible ML preprocessing steps:
  - raw traffic data
  - cleaned daily data
  - feature engineering
  - normalized ML input

## Current DPP Setup

The dashboard currently uses:

- 1 real Mannstaedt DPP
- 2 temporary demo DPPs

The two demo DPPs are included only to provide more traffic data for the prototype and should be replaced once additional Mannstaedt DPPs become available.

## QR Analytics

QR traffic is identified using the existing URL query parameter:

```text
?source=qr
```

The application compares all DPP traffic with traffic filtered by `query=source=qr`.

This allows the dashboard to calculate QR pageviews, non-QR pageviews, QR traffic share, QR average visit duration, non-QR average visit duration, QR device distribution, and QR region distribution.

## Machine Learning

The dashboard includes an experimental 7-day traffic forecast using a Random Forest regression model.

The pipeline is:

```text
Raw Umami traffic
        ↓
Data cleaning
        ↓
Complete daily timeline
        ↓
Feature engineering
        ↓
Feature normalization
        ↓
Random Forest Regression
        ↓
7-day traffic forecast
```

The model uses lag values, rolling averages, and day-of-week information. The dashboard also displays validation MAE so the current prediction quality can be evaluated.

The ML forecast is currently a proof of concept because the historical company-specific traffic is still limited and the two additional DPPs are temporary demo data.

## Project Structure

```text
mannstaedt_dpp_analytics/
├── app/
│   ├── assets/
│   │   └── gmh_gruppe-logo.svg
│   ├── services/
│   │   ├── analytics_service.py
│   │   ├── ml_service.py
│   │   └── umami_service.py
│   ├── config.py
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

The `.env` file contains credentials and must not be committed to Git.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Open:

```text
http://localhost:8501
```

## Docker

Build the Docker image:

```bash
docker build --no-cache -t mannstaedt-dpp-analytics .
```

Run the container:

```bash
docker run \
  --env-file .env \
  -p 8503:8501 \
  --name mannstaedt-dpp-dashboard \
  mannstaedt-dpp-analytics
```

Open:

```text
http://localhost:8503
```

## Notes

- `.env` is intentionally excluded from Git and Docker images.
- `.venv` is only used for local development.
- The Docker image installs dependencies from `requirements.txt`.
- `scikit-learn` is required for the ML forecast.
- The current company-specific forecast should be treated as experimental until more real Mannstaedt traffic and DPPs are available.

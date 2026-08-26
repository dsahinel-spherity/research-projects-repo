import requests

from app.config import (
    UMAMI_BASE_URL,
    UMAMI_WEBSITE_ID,
    UMAMI_USERNAME,
    UMAMI_PASSWORD,
)


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------

def get_token():
    response = requests.post(
        f"{UMAMI_BASE_URL}/api/auth/login",
        json={
            "username": UMAMI_USERNAME,
            "password": UMAMI_PASSWORD,
        },
        timeout=20,
    )

    response.raise_for_status()

    return response.json()["token"]


def get_headers():
    token = get_token()

    return {
        "Authorization": f"Bearer {token}"
    }


# ---------------------------------------------------------
# Overall statistics
# ---------------------------------------------------------

def get_stats(start_at: int, end_at: int):
    params = {
        "startAt": start_at,
        "endAt": end_at,
    }

    response = requests.get(
        f"{UMAMI_BASE_URL}/api/websites/{UMAMI_WEBSITE_ID}/stats",
        headers=get_headers(),
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# Pageviews and sessions over time
# ---------------------------------------------------------

def get_pageviews(
    start_at: int,
    end_at: int,
    unit: str = "day",
):
    params = {
        "startAt": start_at,
        "endAt": end_at,
        "unit": unit,
    }

    response = requests.get(
        f"{UMAMI_BASE_URL}/api/websites/{UMAMI_WEBSITE_ID}/pageviews",
        headers=get_headers(),
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# Generic metrics endpoint
#
# Examples:
# country
# path
# referrer
# browser
# ---------------------------------------------------------

def get_metrics(
    start_at: int,
    end_at: int,
    metric_type: str,
    page: int = 1,
    limit: int = 10,
):
    params = {
        "startAt": start_at,
        "endAt": end_at,
        "page": page,
        "type": metric_type,
        "limit": limit,
    }

    response = requests.get(
        f"{UMAMI_BASE_URL}/api/websites/{UMAMI_WEBSITE_ID}/metrics",
        headers=get_headers(),
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# Realtime analytics
# ---------------------------------------------------------

def get_realtime():
    response = requests.get(
        f"{UMAMI_BASE_URL}/api/realtime/{UMAMI_WEBSITE_ID}",
        headers=get_headers(),
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# Single session details
# ---------------------------------------------------------

def get_session(session_id: str):
    response = requests.get(
        (
            f"{UMAMI_BASE_URL}/api/websites/"
            f"{UMAMI_WEBSITE_ID}/sessions/{session_id}"
        ),
        headers=get_headers(),
        timeout=20,
    )

    response.raise_for_status()

    return response.json()
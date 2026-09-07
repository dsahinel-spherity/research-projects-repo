import requests

from app.config import (
    UMAMI_BASE_URL,
    UMAMI_WEBSITE_ID,
    UMAMI_USERNAME,
    UMAMI_PASSWORD,
)


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

    return {
        "Authorization": f"Bearer {get_token()}"
    }


def get_stats(
    start_at,
    end_at,
    path=None,
    query=None,
):

    params = {
        "startAt": start_at,
        "endAt": end_at,
    }

    if path:
        params["path"] = path

    if query:
        params["query"] = query

    response = requests.get(
        f"{UMAMI_BASE_URL}/api/websites/"
        f"{UMAMI_WEBSITE_ID}/stats",
        headers=get_headers(),
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def get_pageviews(
    start_at,
    end_at,
    unit="day",
    path=None,
    query=None,
):

    params = {
        "startAt": start_at,
        "endAt": end_at,
        "unit": unit,
    }

    if path:
        params["path"] = path

    if query:
        params["query"] = query

    response = requests.get(
        f"{UMAMI_BASE_URL}/api/websites/"
        f"{UMAMI_WEBSITE_ID}/pageviews",
        headers=get_headers(),
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def get_metrics(
    start_at,
    end_at,
    metric_type,
    limit=50,
    path=None,
    query=None,
):

    params = {
        "startAt": start_at,
        "endAt": end_at,
        "type": metric_type,
        "limit": limit,
    }

    if path:
        params["path"] = path

    if query:
        params["query"] = query

    response = requests.get(
        f"{UMAMI_BASE_URL}/api/websites/"
        f"{UMAMI_WEBSITE_ID}/metrics",
        headers=get_headers(),
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()
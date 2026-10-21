from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import requests

from app.config import (
    UMAMI_BASE_URL,
    UMAMI_PASSWORD,
    UMAMI_USERNAME,
    UMAMI_WEBSITE_ID,
)


SECTION_EVENT = "dpp-section-view"
DOCUMENT_EVENT = "dpp-document-open"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4

RETRY_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


# =========================================================
# AUTH
# =========================================================

@lru_cache(maxsize=1)
def get_token() -> str:
    """
    Authenticate once and reuse the token.
    """

    last_error = None

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.post(
                f"{UMAMI_BASE_URL}/api/auth/login",
                json={
                    "username":
                        UMAMI_USERNAME,

                    "password":
                        UMAMI_PASSWORD,
                },
                timeout=
                    REQUEST_TIMEOUT,
            )

            if (
                response.status_code
                in RETRY_STATUS_CODES
            ):

                raise requests.HTTPError(
                    (
                        f"Temporary Umami auth error: "
                        f"{response.status_code}"
                    ),
                    response=response,
                )

            response.raise_for_status()

            payload = (
                response.json()
            )

            token = (
                payload.get(
                    "token"
                )
            )

            if not token:

                raise RuntimeError(
                    "Umami authentication returned no token."
                )

            return token

        except (
            requests.RequestException,
            RuntimeError,
        ) as error:

            last_error = (
                error
            )

            if (
                attempt
                >= MAX_RETRIES - 1
            ):

                break

            time.sleep(
                1.5
                * (
                    2 ** attempt
                )
            )

    raise RuntimeError(
        "Unable to authenticate with Umami."
    ) from last_error


def get_headers(
    token=None,
):

    if token is None:

        token = (
            get_token()
        )

    return {
        "Authorization":
            f"Bearer {token}",
    }


# =========================================================
# BASE GET
# =========================================================

def _get(
    path: str,
    params: dict[str, Any] | None = None,
    token: str | None = None,
):

    url = (
        f"{UMAMI_BASE_URL}"
        f"{path}"
    )

    if token is None:

        token = (
            get_token()
        )

    last_error = None

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.get(
                url,
                headers=
                    get_headers(
                        token
                    ),
                params=
                    params,
                timeout=
                    REQUEST_TIMEOUT,
            )

            # ---------------------------------------------
            # TOKEN EXPIRED
            # ---------------------------------------------

            if (
                response.status_code
                == 401
            ):

                get_token.cache_clear()

                token = (
                    get_token()
                )

                response = requests.get(
                    url,
                    headers=
                        get_headers(
                            token
                        ),
                    params=
                        params,
                    timeout=
                        REQUEST_TIMEOUT,
                )

            # ---------------------------------------------
            # TEMPORARY SERVER FAILURE
            # ---------------------------------------------

            if (
                response.status_code
                in RETRY_STATUS_CODES
            ):

                raise requests.HTTPError(
                    (
                        f"Temporary Umami error "
                        f"{response.status_code}: {url}"
                    ),
                    response=response,
                )

            response.raise_for_status()

            return (
                response.json()
            )

        except (
            requests.RequestException
        ) as error:

            last_error = (
                error
            )

            if (
                attempt
                >= MAX_RETRIES - 1
            ):

                break

            time.sleep(
                1.0
                * (
                    2 ** attempt
                )
            )

    raise RuntimeError(
        (
            f"Umami request failed after "
            f"{MAX_RETRIES} attempts: {url}"
        )
    ) from last_error


# =========================================================
# STATS
# =========================================================

def get_stats(
    start_at,
    end_at,
    path=None,
    query=None,
    token=None,
):

    params = {
        "startAt":
            start_at,

        "endAt":
            end_at,
    }

    if path:

        params[
            "path"
        ] = path

    if query:

        params[
            "query"
        ] = query

    return _get(
        (
            f"/api/websites/"
            f"{UMAMI_WEBSITE_ID}/stats"
        ),
        params=
            params,
        token=
            token,
    )


# =========================================================
# PAGEVIEWS
# =========================================================

def get_pageviews(
    start_at,
    end_at,
    unit="day",
    path=None,
    query=None,
    token=None,
):

    params = {
        "startAt":
            start_at,

        "endAt":
            end_at,

        "unit":
            unit,
    }

    if path:

        params[
            "path"
        ] = path

    if query:

        params[
            "query"
        ] = query

    return _get(
        (
            f"/api/websites/"
            f"{UMAMI_WEBSITE_ID}/pageviews"
        ),
        params=
            params,
        token=
            token,
    )


# =========================================================
# METRICS
# =========================================================

def get_metrics(
    start_at,
    end_at,
    metric_type,
    limit=100,
    path=None,
    query=None,
    token=None,
):

    params = {
        "startAt":
            start_at,

        "endAt":
            end_at,

        "type":
            metric_type,

        "limit":
            limit,
    }

    if path:

        params[
            "path"
        ] = path

    if query:

        params[
            "query"
        ] = query

    return _get(
        (
            f"/api/websites/"
            f"{UMAMI_WEBSITE_ID}/metrics"
        ),
        params=
            params,
        token=
            token,
    )


# =========================================================
# GENERIC PAGINATION
# =========================================================

def _extract_paginated_items(
    payload,
):

    if isinstance(
        payload,
        list,
    ):

        return payload

    if not isinstance(
        payload,
        dict,
    ):

        return []

    for key in [
        "data",
        "items",
        "events",
        "sessions",
    ]:

        value = (
            payload.get(
                key
            )
        )

        if isinstance(
            value,
            list,
        ):

            return value

    return []


def _get_paginated(
    path,
    start_at,
    end_at,
    page_size=100,
    token=None,
):

    results = []

    page = 1

    while True:

        payload = _get(
            path,
            params={
                "startAt":
                    start_at,

                "endAt":
                    end_at,

                "page":
                    page,

                "pageSize":
                    page_size,
            },
            token=
                token,
        )

        items = (
            _extract_paginated_items(
                payload
            )
        )

        if not items:

            break

        results.extend(
            items
        )

        count = None

        if isinstance(
            payload,
            dict,
        ):

            count = (
                payload.get(
                    "count"
                )
            )

        # ---------------------------------------------
        # WE HAVE ALL ROWS
        # ---------------------------------------------

        if (
            count is not None
            and len(
                results
            )
            >= int(
                count
            )
        ):

            break

        if (
            len(
                items
            )
            < page_size
        ):

            break

        page += 1

    return results


# =========================================================
# RAW EVENTS
# =========================================================

def get_events(
    start_at,
    end_at,
    token=None,
):

    return _get_paginated(
        (
            f"/api/websites/"
            f"{UMAMI_WEBSITE_ID}/events"
        ),
        start_at,
        end_at,
        page_size=100,
        token=
            token,
    )


def get_custom_events(
    start_at,
    end_at,
):

    token = (
        get_token()
    )

    events = (
        get_events(
            start_at,
            end_at,
            token=
                token,
        )
    )

    return [
        event
        for event
        in events
        if (
            event.get(
                "eventName"
            )
            in {
                SECTION_EVENT,
                DOCUMENT_EVENT,
            }
        )
    ]


# =========================================================
# EVENT-DATA PIVOT
# =========================================================

def _decode_pivot_properties(
    row,
):
    """
    Convert:

        propertyKeys:
            [
                "dpp_id",
                "organization",
                "section",
                "source",
            ]

        propertyValues:
            [
                "...",
                "...",
                "...",
                "web",
            ]

    into normal dictionary columns.
    """

    keys = (
        row.get(
            "propertyKeys"
        )
        or []
    )

    values = (
        row.get(
            "propertyValues"
        )
        or []
    )

    properties = {}

    for key, value in zip(
        keys,
        values,
    ):

        if key:

            properties[
                str(
                    key
                )
            ] = value

    return properties


@lru_cache(maxsize=128)
def _get_event_data_pivot_cached(
    start_at,
    end_at,
    event_name,
):
    """
    Internal cache for pivot data.

    This avoids downloading the same historical custom-event
    metadata repeatedly while Streamlit is running.
    """

    token = (
        get_token()
    )

    return tuple(
        _get_event_data_pivot_uncached(
            start_at,
            end_at,
            event_name,
            token=
                token,
        )
    )


def get_event_data_pivot(
    start_at,
    end_at,
    event_name,
    token=None,
):
    """
    Public event-data pivot reader.

    If no explicit token is supplied, use the cached version.
    """

    if token is None:

        return [
            dict(
                row
            )
            for row
            in _get_event_data_pivot_cached(
                int(
                    start_at
                ),
                int(
                    end_at
                ),
                str(
                    event_name
                ),
            )
        ]

    return (
        _get_event_data_pivot_uncached(
            start_at,
            end_at,
            event_name,
            token=
                token,
        )
    )


def _get_event_data_pivot_uncached(
    start_at,
    end_at,
    event_name,
    token=None,
    page_size=250,
):

    if token is None:

        token = (
            get_token()
        )

    rows = []

    page = 1

    while True:

        payload = _get(
            (
                f"/api/websites/"
                f"{UMAMI_WEBSITE_ID}/"
                f"event-data-pivot"
            ),
            params={
                "startAt":
                    start_at,

                "endAt":
                    end_at,

                # IMPORTANT:
                # Your Umami instance expects eventName,
                # NOT event.
                "eventName":
                    event_name,

                "page":
                    page,

                "pageSize":
                    page_size,
            },
            token=
                token,
        )

        if not isinstance(
            payload,
            dict,
        ):

            break

        data = (
            payload.get(
                "data"
            )
            or []
        )

        if not data:

            break

        for raw_row in data:

            if not isinstance(
                raw_row,
                dict,
            ):

                continue

            row = (
                raw_row.copy()
            )

            properties = (
                _decode_pivot_properties(
                    row
                )
            )

            row.update(
                properties
            )

            # Normalize event ID to same field used by /events.
            if (
                "eventId"
                in row
            ):

                row[
                    "id"
                ] = (
                    row[
                        "eventId"
                    ]
                )

            rows.append(
                row
            )

        count = (
            payload.get(
                "count"
            )
        )

        if (
            count is not None
            and len(
                rows
            )
            >= int(
                count
            )
        ):

            break

        if (
            len(
                data
            )
            < page_size
        ):

            break

        page += 1

    return rows


# =========================================================
# BACKWARD-COMPATIBILITY HELPER
# =========================================================

def get_event_data(
    event_id,
):
    """
    Kept only so older diagnostics/imports do not crash.

    The current Umami API does not provide reliable row-level
    custom data via /event-data/{eventId}; we now use the
    event-data-pivot endpoint instead.
    """

    return {}


# =========================================================
# ENRICH EVENTS WITH PIVOT DATA
# =========================================================

def get_events_with_data(
    start_at,
    end_at,
    token=None,
):

    if token is None:

        token = (
            get_token()
        )

    # ---------------------------------------------
    # RAW EVENTS
    # ---------------------------------------------

    raw_events = (
        get_events(
            start_at,
            end_at,
            token=
                token,
        )
    )

    if not raw_events:

        return []

    # ---------------------------------------------
    # LOAD BOTH CUSTOM EVENT TYPES IN PARALLEL
    # ---------------------------------------------

    pivot_results = {}

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:

        futures = {
            executor.submit(
                _get_event_data_pivot_uncached,
                start_at,
                end_at,
                SECTION_EVENT,
                token,
            ):
                SECTION_EVENT,

            executor.submit(
                _get_event_data_pivot_uncached,
                start_at,
                end_at,
                DOCUMENT_EVENT,
                token,
            ):
                DOCUMENT_EVENT,
        }

        for future in (
            as_completed(
                futures
            )
        ):

            event_name = (
                futures[
                    future
                ]
            )

            pivot_results[
                event_name
            ] = (
                future.result()
            )

    section_rows = (
        pivot_results.get(
            SECTION_EVENT,
            [],
        )
    )

    document_rows = (
        pivot_results.get(
            DOCUMENT_EVENT,
            [],
        )
    )

    pivot_rows = (
        section_rows
        + document_rows
    )

    # ---------------------------------------------
    # LOOKUP BY EVENT ID
    # ---------------------------------------------

    pivot_lookup = {}

    for row in pivot_rows:

        event_id = (
            row.get(
                "eventId"
            )
            or row.get(
                "id"
            )
        )

        if not event_id:

            continue

        pivot_lookup[
            str(
                event_id
            )
        ] = row

    # ---------------------------------------------
    # MERGE CUSTOM DATA INTO /events ROWS
    # ---------------------------------------------

    enriched = []

    for raw_event in raw_events:

        row = (
            raw_event.copy()
        )

        event_id = (
            row.get(
                "id"
            )
        )

        if event_id:

            pivot_row = (
                pivot_lookup.get(
                    str(
                        event_id
                    )
                )
            )

            if pivot_row:

                # Only copy custom property fields here.
                # Preserve canonical event fields from /events.

                for key in [
                    "dpp_id",
                    "organization",
                    "section",
                    "document",
                    "source",
                ]:

                    if (
                        key
                        in pivot_row
                    ):

                        row[
                            key
                        ] = (
                            pivot_row[
                                key
                            ]
                        )

        enriched.append(
            row
        )

    return enriched


# =========================================================
# SESSIONS
# =========================================================

def get_sessions(
    start_at,
    end_at,
    token=None,
):

    return _get_paginated(
        (
            f"/api/websites/"
            f"{UMAMI_WEBSITE_ID}/sessions"
        ),
        start_at,
        end_at,
        page_size=100,
        token=
            token,
    )


def get_session(
    session_id,
):

    token = (
        get_token()
    )

    return _get(
        (
            f"/api/websites/"
            f"{UMAMI_WEBSITE_ID}/"
            f"sessions/{session_id}"
        ),
        token=
            token,
    )


def get_session_activity(
    session_id,
    start_at,
    end_at,
):

    token = (
        get_token()
    )

    return _get(
        (
            f"/api/websites/"
            f"{UMAMI_WEBSITE_ID}/"
            f"sessions/{session_id}/activity"
        ),
        params={
            "startAt":
                start_at,

            "endAt":
                end_at,
        },
        token=
            token,
    )


# =========================================================
# COMPLETE DASHBOARD LOAD
# =========================================================

def get_real_analytics_data(
    start_at,
    end_at,
):
    """
    Main real-data loader.

    Authentication happens once.

    Independent API resources are then loaded concurrently.

    Custom event metadata is loaded using event-data-pivot,
    NOT hundreds of /event-data/{eventId} requests.
    """

    token = (
        get_token()
    )

    jobs = {
        "stats":
            lambda:
                get_stats(
                    start_at,
                    end_at,
                    token=
                        token,
                ),

        "events":
            lambda:
                get_events_with_data(
                    start_at,
                    end_at,
                    token=
                        token,
                ),

        "sessions":
            lambda:
                get_sessions(
                    start_at,
                    end_at,
                    token=
                        token,
                ),

        "pageviews":
            lambda:
                get_pageviews(
                    start_at,
                    end_at,
                    token=
                        token,
                ),

        "devices":
            lambda:
                get_metrics(
                    start_at,
                    end_at,
                    "device",
                    token=
                        token,
                ),

        "countries":
            lambda:
                get_metrics(
                    start_at,
                    end_at,
                    "country",
                    token=
                        token,
                ),

        "regions":
            lambda:
                get_metrics(
                    start_at,
                    end_at,
                    "region",
                    token=
                        token,
                ),

        "paths":
            lambda:
                get_metrics(
                    start_at,
                    end_at,
                    "path",
                    token=
                        token,
                ),
    }

    results = {}

    # Moderate concurrency:
    # enough to be fast, but not enough to hammer non-prod.
    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:

        futures = {
            executor.submit(
                function
            ):
                name

            for name, function
            in jobs.items()
        }

        for future in (
            as_completed(
                futures
            )
        ):

            name = (
                futures[
                    future
                ]
            )

            results[
                name
            ] = (
                future.result()
            )

    return results


# =========================================================
# CACHE CONTROL
# =========================================================

def clear_umami_cache():
    """
    Clear pivot-data cache.

    We intentionally keep the authentication token cached
    during normal dashboard refreshes to avoid unnecessary
    /api/auth/login calls.
    """

    _get_event_data_pivot_cached.cache_clear()


def clear_auth_cache():
    """
    Only needed if authentication itself needs resetting.
    """

    get_token.cache_clear()


def clear_event_cache():
    """
    Alias for clearing cached pivot event data.
    """

    _get_event_data_pivot_cached.cache_clear()
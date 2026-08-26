import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.services.umami_service import (
    get_stats,
    get_pageviews,
    get_metrics,
    get_realtime,
    get_session,
)

from app.services.analytics_service import (
    calculate_overview_with_comparison,
    normalize_metrics,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VERA DPP Analytics",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# FILE PATHS
# =========================================================

LOGO_PATH = (
    PROJECT_ROOT
    / "app"
    / "assets"
    / "vera_logo.png"
)


# =========================================================
# CACHED DATA LOADERS
# =========================================================

@st.cache_data(
    ttl=60,
    show_spinner=False
)
def load_stats(
    start_at: int,
    end_at: int
):
    return get_stats(
        start_at,
        end_at
    )


@st.cache_data(
    ttl=60,
    show_spinner=False
)
def load_pageviews(
    start_at: int,
    end_at: int,
    unit: str = "day"
):
    return get_pageviews(
        start_at,
        end_at,
        unit=unit
    )


@st.cache_data(
    ttl=60,
    show_spinner=False
)
def load_metrics(
    start_at: int,
    end_at: int,
    metric_type: str,
    limit: int = 10
):
    return get_metrics(
        start_at,
        end_at,
        metric_type=metric_type,
        limit=limit
    )


@st.cache_data(
    ttl=30,
    show_spinner=False
)
def load_realtime():
    return get_realtime()


@st.cache_data(
    ttl=60,
    show_spinner=False
)
def load_session(
    session_id: str
):
    return get_session(
        session_id
    )


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def datetime_to_ms(dt):
    """
    Convert datetime to Unix milliseconds.
    Umami expects startAt and endAt in milliseconds.
    """
    return int(
        dt.timestamp() * 1000
    )


def seconds_to_readable(seconds):
    """
    Convert seconds to a readable duration.

    Example:
    367 seconds -> 6m 7s
    """

    seconds = int(seconds)

    minutes = seconds // 60
    remaining_seconds = (
        seconds % 60
    )

    if minutes == 0:
        return (
            f"{remaining_seconds}s"
        )

    return (
        f"{minutes}m "
        f"{remaining_seconds}s"
    )


def metrics_to_dataframe(
    data,
    name_column="name"
):
    """
    Convert normalized metric data into a DataFrame.
    """

    if not data:
        return pd.DataFrame(
            columns=[
                name_column,
                "count"
            ]
        )

    df = pd.DataFrame(data)

    df = df.rename(
        columns={
            "name": name_column
        }
    )

    return df


def shorten_dpp_path(path):
    """
    Make long DPP DID paths easier to read
    in the dashboard.

    Example:
    /did:web:api.vera...:battery-123
    -> battery-123
    """

    if not isinstance(path, str):
        return path

    if ":did-registry:" in path:
        return path.split(
            ":did-registry:"
        )[-1]

    return path


# =========================================================
# SIDEBAR
# =========================================================

if LOGO_PATH.exists():

    st.sidebar.image(
        str(LOGO_PATH),
        use_container_width=True
    )


st.sidebar.title(
    "DPP Analytics"
)

st.sidebar.caption(
    "VERA by SPHERITY"
)


date_option = (
    st.sidebar.selectbox(
        "Date range",
        [
            "Last 7 days",
            "Last 30 days",
            "Custom"
        ]
    )
)


now = datetime.now(
    timezone.utc
)


if date_option == "Last 7 days":

    start_date = (
        now
        - timedelta(days=7)
    )

    end_date = now


elif date_option == "Last 30 days":

    start_date = (
        now
        - timedelta(days=30)
    )

    end_date = now


else:

    custom_start = (
        st.sidebar.date_input(
            "Start date",
            value=(
                now
                - timedelta(days=7)
            ).date()
        )
    )

    custom_end = (
        st.sidebar.date_input(
            "End date",
            value=now.date()
        )
    )

    start_date = datetime.combine(
        custom_start,
        datetime.min.time(),
        tzinfo=timezone.utc
    )

    end_date = datetime.combine(
        custom_end,
        datetime.max.time(),
        tzinfo=timezone.utc
    )


start_at = datetime_to_ms(
    start_date
)

end_at = datetime_to_ms(
    end_date
)


# =========================================================
# REFRESH BUTTON
# =========================================================

if st.sidebar.button(
    "Refresh data"
):

    st.cache_data.clear()

    st.rerun()


# =========================================================
# HEADER
# =========================================================

header_col1, header_col2 = (
    st.columns(
        [1.2, 4]
    )
)


with header_col1:

    if LOGO_PATH.exists():

        st.image(
            str(LOGO_PATH),
            width=260
        )


with header_col2:

    st.title(
        "DPP Analytics Dashboard"
    )

    st.caption(
        "User behaviour analytics for "
        "Vera Digital Product Passports"
    )


st.caption(
    "Last refreshed: "
    + datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)


# =========================================================
# LOAD MAIN UMAMI DATA
# =========================================================

try:

    raw_stats = load_stats(
        start_at,
        end_at
    )

    overview = (
        calculate_overview_with_comparison(
            raw_stats
        )
    )

    traffic = load_pageviews(
        start_at,
        end_at,
        unit="day"
    )

    paths_raw = load_metrics(
        start_at,
        end_at,
        metric_type="path",
        limit=20
    )

    countries_raw = load_metrics(
        start_at,
        end_at,
        metric_type="country",
        limit=10
    )

    browsers_raw = load_metrics(
        start_at,
        end_at,
        metric_type="browser",
        limit=10
    )

    referrers_raw = load_metrics(
        start_at,
        end_at,
        metric_type="referrer",
        limit=10
    )


except Exception as error:

    st.error(
        "Could not load Umami data: "
        f"{error}"
    )

    st.stop()


# =========================================================
# KPI CARDS
# =========================================================

current = overview[
    "current"
]

changes = overview[
    "changes_percent"
]


col1, col2, col3, col4, col5 = (
    st.columns(5)
)


with col1:

    st.metric(
        "Pageviews",
        current["pageviews"],
        (
            f'{changes["pageviews"]}%'
            if changes["pageviews"]
            is not None
            else None
        )
    )


with col2:

    st.metric(
        "Visitors",
        current["visitors"],
        (
            f'{changes["visitors"]}%'
            if changes["visitors"]
            is not None
            else None
        )
    )


with col3:

    st.metric(
        "Visits",
        current["visits"],
        (
            f'{changes["visits"]}%'
            if changes["visits"]
            is not None
            else None
        )
    )


with col4:

    st.metric(
        "Bounce Rate",
        (
            f'{current["bounce_rate_percent"]}%'
        )
    )


with col5:

    st.metric(
        "Avg Visit Duration",
        seconds_to_readable(
            current[
                "avg_visit_duration_seconds"
            ]
        )
    )


st.divider()


# =========================================================
# TRAFFIC OVER TIME
# =========================================================

st.subheader(
    "Traffic Over Time"
)


pageviews_df = pd.DataFrame(
    traffic.get(
        "pageviews",
        []
    )
)

sessions_df = pd.DataFrame(
    traffic.get(
        "sessions",
        []
    )
)


if not pageviews_df.empty:

    pageviews_df = (
        pageviews_df.rename(
            columns={
                "x": "date",
                "y": "pageviews"
            }
        )
    )

    pageviews_df[
        "date"
    ] = pd.to_datetime(
        pageviews_df["date"]
    )


if not sessions_df.empty:

    sessions_df = (
        sessions_df.rename(
            columns={
                "x": "date",
                "y": "sessions"
            }
        )
    )

    sessions_df[
        "date"
    ] = pd.to_datetime(
        sessions_df["date"]
    )


if (
    not pageviews_df.empty
    and not sessions_df.empty
):

    traffic_df = pd.merge(
        pageviews_df,
        sessions_df,
        on="date",
        how="outer"
    )

    traffic_df = (
        traffic_df.fillna(0)
    )

    traffic_df = (
        traffic_df.set_index(
            "date"
        )
    )

    st.line_chart(
        traffic_df[
            [
                "pageviews",
                "sessions"
            ]
        ]
    )


else:

    st.info(
        "No traffic data available "
        "for this date range."
    )


st.divider()


# =========================================================
# MOST VIEWED DPP PAGES
# =========================================================

st.subheader(
    "Most Viewed DPP Pages"
)


paths = normalize_metrics(
    paths_raw
)


paths_df = metrics_to_dataframe(
    paths,
    name_column="path"
)


if not paths_df.empty:

    # Only keep actual DPP paths.
    # This removes /login and similar pages.
    paths_df = paths_df[
        paths_df["path"]
        .str.startswith(
            "/did:web:",
            na=False
        )
    ].copy()

    paths_df[
        "dpp"
    ] = paths_df[
        "path"
    ].apply(
        shorten_dpp_path
    )

    paths_df = (
        paths_df.sort_values(
            "count",
            ascending=False
        )
    )


if not paths_df.empty:

    chart_col, table_col = (
        st.columns(
            [2, 1]
        )
    )


    with chart_col:

        chart_df = (
            paths_df[
                [
                    "dpp",
                    "count"
                ]
            ]
            .set_index("dpp")
        )

        st.bar_chart(
            chart_df[
                "count"
            ]
        )


    with table_col:

        st.dataframe(
            paths_df[
                [
                    "dpp",
                    "count"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


else:

    st.info(
        "No DPP path data available."
    )


st.divider()


# =========================================================
# COUNTRY AND BROWSER ANALYTICS
# =========================================================

left, right = st.columns(2)


# ---------------------------------------------------------
# Countries
# ---------------------------------------------------------

with left:

    st.subheader(
        "Visitors by Country"
    )

    countries = (
        normalize_metrics(
            countries_raw
        )
    )

    countries_df = (
        metrics_to_dataframe(
            countries,
            name_column="country"
        )
    )


    if not countries_df.empty:

        countries_df = (
            countries_df.sort_values(
                "count",
                ascending=False
            )
        )

        st.bar_chart(
            countries_df
            .set_index(
                "country"
            )[
                "count"
            ]
        )

        st.dataframe(
            countries_df,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.info(
            "No country data available."
        )


# ---------------------------------------------------------
# Browsers
# ---------------------------------------------------------

with right:

    st.subheader(
        "Visitors by Browser"
    )

    browsers = (
        normalize_metrics(
            browsers_raw
        )
    )

    browsers_df = (
        metrics_to_dataframe(
            browsers,
            name_column="browser"
        )
    )


    if not browsers_df.empty:

        browsers_df = (
            browsers_df.sort_values(
                "count",
                ascending=False
            )
        )

        st.bar_chart(
            browsers_df
            .set_index(
                "browser"
            )[
                "count"
            ]
        )

        st.dataframe(
            browsers_df,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.info(
            "No browser data available."
        )


st.divider()


# =========================================================
# REFERRERS
# =========================================================

st.subheader(
    "Top Referrers"
)


referrers = (
    normalize_metrics(
        referrers_raw
    )
)


referrers_df = (
    metrics_to_dataframe(
        referrers,
        name_column="referrer"
    )
)


if not referrers_df.empty:

    referrers_df = (
        referrers_df.sort_values(
            "count",
            ascending=False
        )
    )

    referrer_col1, referrer_col2 = (
        st.columns(
            [2, 1]
        )
    )


    with referrer_col1:

        st.bar_chart(
            referrers_df
            .set_index(
                "referrer"
            )[
                "count"
            ]
        )


    with referrer_col2:

        st.dataframe(
            referrers_df,
            use_container_width=True,
            hide_index=True
        )


else:

    st.info(
        "No referrer data available."
    )


st.divider()


# =========================================================
# REALTIME
# =========================================================

st.subheader(
    "Realtime Activity"
)


try:

    realtime = (
        load_realtime()
    )

    totals = realtime.get(
        "totals",
        {}
    )


    r1, r2, r3, r4 = (
        st.columns(4)
    )


    r1.metric(
        "Views",
        totals.get(
            "views",
            0
        )
    )

    r2.metric(
        "Visitors",
        totals.get(
            "visitors",
            0
        )
    )

    r3.metric(
        "Events",
        totals.get(
            "events",
            0
        )
    )

    r4.metric(
        "Countries",
        totals.get(
            "countries",
            0
        )
    )


    realtime_events = (
        realtime.get(
            "events",
            []
        )
    )


    if realtime_events:

        realtime_df = (
            pd.DataFrame(
                realtime_events
            )
        )


        columns_to_show = [
            "__type",
            "createdAt",
            "sessionId",
            "country",
            "device",
            "browser",
            "os",
            "urlPath",
        ]


        available_columns = [
            column
            for column
            in columns_to_show
            if column
            in realtime_df.columns
        ]


        st.dataframe(
            realtime_df[
                available_columns
            ],
            use_container_width=True,
            hide_index=True
        )


    else:

        st.info(
            "No realtime activity "
            "at the moment."
        )


except Exception as error:

    st.warning(
        "Realtime data unavailable: "
        f"{error}"
    )


st.divider()


# =========================================================
# SESSION LOOKUP
# =========================================================

st.subheader(
    "Session Lookup"
)


session_id = st.text_input(
    "Enter a session ID",
    placeholder=(
        "Example: "
        "66fc210c-9eb9-5d8b-b8e1-0642cc5b3bbe"
    )
)


if st.button(
    "Load Session"
):

    if not session_id:

        st.warning(
            "Please enter a session ID."
        )


    else:

        try:

            session_data = (
                load_session(
                    session_id.strip()
                )
            )


            st.success(
                "Session found."
            )


            c1, c2, c3 = (
                st.columns(3)
            )


            c1.metric(
                "Visits",
                session_data.get(
                    "visits",
                    0
                )
            )

            c2.metric(
                "Views",
                session_data.get(
                    "views",
                    0
                )
            )

            c3.metric(
                "Events",
                session_data.get(
                    "events",
                    0
                )
            )


            detail1, detail2 = (
                st.columns(2)
            )


            with detail1:

                st.write(
                    "**Browser:**",
                    session_data.get(
                        "browser"
                    )
                )

                st.write(
                    "**OS:**",
                    session_data.get(
                        "os"
                    )
                )

                st.write(
                    "**Device:**",
                    session_data.get(
                        "device"
                    )
                )

                st.write(
                    "**Screen:**",
                    session_data.get(
                        "screen"
                    )
                )


            with detail2:

                st.write(
                    "**Country:**",
                    session_data.get(
                        "country"
                    )
                )

                st.write(
                    "**Region:**",
                    session_data.get(
                        "region"
                    )
                )

                st.write(
                    "**City:**",
                    session_data.get(
                        "city"
                    )
                )

                st.write(
                    "**Language:**",
                    session_data.get(
                        "language"
                    )
                )


            st.write(
                "**First seen:**",
                session_data.get(
                    "firstAt"
                )
            )

            st.write(
                "**Last seen:**",
                session_data.get(
                    "lastAt"
                )
            )


            with st.expander(
                "Raw session data"
            ):

                st.json(
                    session_data
                )


        except Exception as error:

            st.error(
                "Could not load session: "
                f"{error}"
            )


# =========================================================
# FUTURE ANALYTICS
# =========================================================

st.divider()


st.subheader(
    "Future DPP Interaction Analytics"
)


st.info(
    """
Once the missing custom DPP events are implemented,
this dashboard can be extended with:

- QR-entry analysis
- Section engagement
- Document views
- PDF downloads
- QR sessions with vs. without PDF downloads
- User journey and transition analysis
"""
)
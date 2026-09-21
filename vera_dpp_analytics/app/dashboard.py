import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from app.services.umami_service import (
    get_stats,
    get_pageviews,
    get_metrics,
    get_realtime,
    get_session,
)

from app.services.analytics_service import (
    calculate_overview_with_comparison,
)

from app.services.ml_service import (
    build_traffic_forecast,
)


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="VERA DPP Analytics",
    layout="wide",
)


LOGO_PATH = (
    PROJECT_ROOT
    / "app"
    / "assets"
    / "vera_logo.png"
)


# ---------------------------------------------------------
# Cached API functions
# ---------------------------------------------------------

@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def load_stats(
    start_at,
    end_at,
):
    return get_stats(
        start_at,
        end_at,
    )


@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def load_pageviews(
    start_at,
    end_at,
):
    return get_pageviews(
        start_at,
        end_at,
        unit="day",
    )


@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def load_metrics(
    start_at,
    end_at,
    metric_type,
    limit=10,
):
    return get_metrics(
        start_at,
        end_at,
        metric_type,
        limit=limit,
    )


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_realtime():
    return get_realtime()


@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def load_session(
    session_id,
):
    return get_session(
        session_id
    )


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def datetime_to_ms(value):
    return int(
        value.timestamp() * 1000
    )


def seconds_to_readable(seconds):
    seconds = int(seconds)

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return (
        f"{minutes}m "
        f"{remaining_seconds}s"
    )


def shorten_dpp_path(path):
    if not path:
        return "Unknown"

    if ":did-registry:" in path:
        return path.split(
            ":did-registry:"
        )[-1]

    return path


def metric_delta(value):
    if value is None:
        return None

    return f"{value:+.2f}%"


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            width="stretch",
        )

    st.title(
        "DPP Analytics"
    )

    st.caption(
        "VERA by SPHERITY"
    )

    st.divider()

    period = st.selectbox(
        "Date range",
        [
            "Last 7 days",
            "Last 30 days",
            "Custom",
        ],
        index=1,
    )

    now = datetime.now(
        timezone.utc
    )

    if period == "Last 7 days":

        start_date = (
            now
            - timedelta(days=7)
        )

        end_date = now

    elif period == "Last 30 days":

        start_date = (
            now
            - timedelta(days=30)
        )

        end_date = now

    else:

        dates = st.date_input(
            "Select date range",
            value=(
                (
                    now
                    - timedelta(days=30)
                ).date(),
                now.date(),
            ),
        )

        if (
            isinstance(dates, tuple)
            and len(dates) == 2
        ):
            start_date = datetime.combine(
                dates[0],
                datetime.min.time(),
                tzinfo=timezone.utc,
            )

            end_date = datetime.combine(
                dates[1],
                datetime.max.time(),
                tzinfo=timezone.utc,
            )

        else:
            start_date = (
                now
                - timedelta(days=30)
            )

            end_date = now

    st.divider()

    if st.button(
        "Refresh Data",
        width="stretch",
    ):
        st.cache_data.clear()
        st.rerun()


start_at = datetime_to_ms(
    start_date
)

end_at = datetime_to_ms(
    end_date
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

header_logo, header_title = (
    st.columns([1, 5])
)


with header_logo:

    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            width="stretch",
        )


with header_title:

    st.title(
        "VERA DPP Analytics Dashboard"
    )

    st.caption(
        "Analytics based on Umami data"
    )


st.caption(
    "Last refreshed: "
    + datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)


st.divider()


# ---------------------------------------------------------
# Load main data
# ---------------------------------------------------------

try:

    stats = load_stats(
        start_at,
        end_at,
    )

    pageviews_data = load_pageviews(
        start_at,
        end_at,
    )

    paths = load_metrics(
        start_at,
        end_at,
        "path",
        20,
    )

    countries = load_metrics(
        start_at,
        end_at,
        "country",
        10,
    )

    browsers = load_metrics(
        start_at,
        end_at,
        "browser",
        10,
    )

    referrers = load_metrics(
        start_at,
        end_at,
        "referrer",
        10,
    )

except Exception as e:

    st.error(
        f"Could not load Umami data: {e}"
    )

    st.stop()


overview = (
    calculate_overview_with_comparison(
        stats
    )
)

current = overview["current"]

changes = overview[
    "changes_percent"
]


# ---------------------------------------------------------
# KPI overview
# ---------------------------------------------------------

st.subheader(
    "Overview"
)


kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)


kpi1.metric(
    "Pageviews",
    current["pageviews"],
    metric_delta(
        changes["pageviews"]
    ),
)


kpi2.metric(
    "Visitors",
    current["visitors"],
    metric_delta(
        changes["visitors"]
    ),
)


kpi3.metric(
    "Visits",
    current["visits"],
    metric_delta(
        changes["visits"]
    ),
)


kpi4.metric(
    "Bounce Rate",
    f"{current['bounce_rate_percent']}%",
)


kpi5.metric(
    "Avg Visit Duration",
    seconds_to_readable(
        current[
            "avg_visit_duration_seconds"
        ]
    ),
)


st.divider()


# ---------------------------------------------------------
# Traffic over time
# ---------------------------------------------------------

st.subheader(
    "Traffic Over Time"
)


pageview_series = (
    pageviews_data.get(
        "pageviews",
        []
    )
)

session_series = (
    pageviews_data.get(
        "sessions",
        []
    )
)


if pageview_series:

    pv_df = pd.DataFrame(
        pageview_series
    )

    pv_df["date"] = pd.to_datetime(
        pv_df["x"]
    )

    pv_df = pv_df.rename(
        columns={
            "y": "Pageviews"
        }
    )

    traffic_df = pv_df[
        [
            "date",
            "Pageviews",
        ]
    ]


    if session_series:

        sessions_df = pd.DataFrame(
            session_series
        )

        sessions_df[
            "date"
        ] = pd.to_datetime(
            sessions_df["x"]
        )

        sessions_df = (
            sessions_df.rename(
                columns={
                    "y": "Sessions"
                }
            )
        )

        traffic_df = traffic_df.merge(
            sessions_df[
                [
                    "date",
                    "Sessions",
                ]
            ],
            on="date",
            how="left",
        )


    st.line_chart(
        traffic_df.set_index(
            "date"
        ),
        width="stretch",
    )


else:

    st.info(
        "No traffic data found."
    )


st.divider()


# ---------------------------------------------------------
# Most viewed DPP pages
# ---------------------------------------------------------

st.subheader(
    "Most Viewed DPP Pages"
)


dpp_paths = [
    item
    for item in paths
    if str(
        item.get("x", "")
    ).startswith(
        "/did:web:"
    )
]


if dpp_paths:

    dpp_df = pd.DataFrame(
        dpp_paths
    )

    dpp_df["DPP"] = (
        dpp_df["x"]
        .apply(
            shorten_dpp_path
        )
    )

    dpp_df = dpp_df.rename(
        columns={
            "y": "Views"
        }
    )


    st.bar_chart(
        dpp_df[
            [
                "DPP",
                "Views",
            ]
        ].set_index("DPP"),
        width="stretch",
    )


    st.dataframe(
        dpp_df[
            [
                "DPP",
                "Views",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


else:

    st.info(
        "No DPP pageviews found."
    )


st.divider()


# ---------------------------------------------------------
# Country and browser
# ---------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Top Countries"
    )

    if countries:

        country_df = pd.DataFrame(
            countries
        ).rename(
            columns={
                "x": "Country",
                "y": "Visitors",
            }
        )

        st.bar_chart(
            country_df.set_index(
                "Country"
            ),
            width="stretch",
        )

        st.dataframe(
            country_df,
            width="stretch",
            hide_index=True,
        )


with col2:

    st.subheader(
        "Top Browsers"
    )

    if browsers:

        browser_df = pd.DataFrame(
            browsers
        ).rename(
            columns={
                "x": "Browser",
                "y": "Visitors",
            }
        )

        st.bar_chart(
            browser_df.set_index(
                "Browser"
            ),
            width="stretch",
        )

        st.dataframe(
            browser_df,
            width="stretch",
            hide_index=True,
        )


st.divider()


# ---------------------------------------------------------
# Referrers
# ---------------------------------------------------------

st.subheader(
    "Top Referrers"
)


if referrers:

    referrer_df = pd.DataFrame(
        referrers
    ).rename(
        columns={
            "x": "Referrer",
            "y": "Count",
        }
    )

    st.bar_chart(
        referrer_df.set_index(
            "Referrer"
        ),
        width="stretch",
    )

    st.dataframe(
        referrer_df,
        width="stretch",
        hide_index=True,
    )


st.divider()


# ---------------------------------------------------------
# Machine learning forecast
# ---------------------------------------------------------

st.subheader(
    "ML Traffic Forecast"
)

st.caption(
    "Experimental prediction of VERA pageviews for the next 7 days."
)


forecast_result = (
    build_traffic_forecast(
        pageviews_data,
        days=7,
    )
)


if (
    forecast_result["status"]
    == "not_enough_data"
):

    st.warning(
        "Not enough historical daily traffic data "
        "to train the forecasting model yet."
    )

else:

    historical_df = (
        forecast_result[
            "historical"
        ].copy()
    )

    forecast_df = (
        forecast_result[
            "forecast"
        ].copy()
    )


    ml1, ml2, ml3 = (
        st.columns(3)
    )


    predicted_total = int(
        forecast_df[
            "predicted_pageviews"
        ].sum()
    )


    avg_predicted = round(
        forecast_df[
            "predicted_pageviews"
        ].mean(),
        1,
    )


    ml1.metric(
        "Predicted Pageviews",
        predicted_total,
        help="Predicted total pageviews for the next 7 days.",
    )


    ml2.metric(
        "Predicted Daily Average",
        avg_predicted,
    )


    if (
        forecast_result["mae"]
        is not None
    ):

        ml3.metric(
            "Validation MAE",
            forecast_result[
                "mae"
            ],
            help=(
                "Mean Absolute Error on a small historical "
                "holdout period. Lower is better."
            ),
        )

    else:

        ml3.metric(
            "Validation MAE",
            "N/A",
        )


    actual_chart = (
        historical_df[
            [
                "date",
                "pageviews",
            ]
        ]
        .rename(
            columns={
                "pageviews": "Actual"
            }
        )
    )


    prediction_chart = (
        forecast_df[
            [
                "date",
                "predicted_pageviews",
            ]
        ]
        .rename(
            columns={
                "predicted_pageviews":
                    "Predicted"
            }
        )
    )


    combined_chart = (
        actual_chart.merge(
            prediction_chart,
            on="date",
            how="outer",
        )
        .sort_values("date")
        .set_index("date")
    )


    st.line_chart(
        combined_chart,
        width="stretch",
    )


    display_forecast = (
        forecast_df.copy()
    )

    display_forecast[
        "date"
    ] = display_forecast[
        "date"
    ].dt.date


    display_forecast = (
        display_forecast.rename(
            columns={
                "date": "Date",
                "predicted_pageviews":
                    "Predicted Pageviews",
            }
        )
    )


    st.dataframe(
        display_forecast,
        width="stretch",
        hide_index=True,
    )


    st.info(
        "This is a prototype forecast based only on "
        "historical Umami pageview patterns. "
        "With more historical data, the model can be "
        "evaluated and improved."
    )


st.divider()


# ---------------------------------------------------------
# Realtime
# ---------------------------------------------------------

st.subheader(
    "Realtime Activity"
)


try:

    realtime = load_realtime()

    totals = realtime.get(
        "totals",
        {}
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Realtime Visitors",
        totals.get(
            "visitors",
            0,
        ),
    )

    r2.metric(
        "Realtime Pageviews",
        totals.get(
            "pageviews",
            0,
        ),
    )

    r3.metric(
        "Realtime Events",
        totals.get(
            "events",
            0,
        ),
    )


    realtime_events = (
        realtime.get(
            "events",
            []
        )
    )


    if realtime_events:

        realtime_df = pd.DataFrame(
            realtime_events
        )

        st.dataframe(
            realtime_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.caption(
            "No realtime events currently."
        )


except Exception as e:

    st.warning(
        f"Realtime data unavailable: {e}"
    )


st.divider()


# ---------------------------------------------------------
# Session lookup
# ---------------------------------------------------------

st.subheader(
    "Session Lookup"
)


session_id = st.text_input(
    "Enter Umami session ID"
)


if session_id:

    try:

        session = load_session(
            session_id
        )


        s1, s2, s3 = st.columns(3)


        s1.metric(
            "Visits",
            session.get(
                "visits",
                0,
            ),
        )


        s2.metric(
            "Views",
            session.get(
                "views",
                0,
            ),
        )


        s3.metric(
            "Events",
            session.get(
                "events",
                0,
            ),
        )


        st.json(session)


    except Exception as e:

        st.error(
            f"Could not load session: {e}"
        )


st.divider()


# ---------------------------------------------------------
# Future DPP analytics
# ---------------------------------------------------------

st.subheader(
    "Future DPP Analytics"
)


st.info(
    """
Once additional VERA interaction events are tracked,
this dashboard can be extended with:

- QR entry analytics
- Section engagement
- Document views
- PDF downloads
- QR-to-download conversion
- User journey / transition analysis
"""
)
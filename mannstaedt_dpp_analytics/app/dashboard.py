import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORT SERVICES
# =========================================================

from app.services.umami_service import (
    get_stats,
    get_pageviews,
    get_metrics,
)

from app.services.analytics_service import (
    calculate_qr_share,
    aggregate_dpp_stats,
    aggregate_timeseries,
    aggregate_metric_lists,
    calculate_non_qr_pageviews,
)

from app.services.ml_service import (
    build_company_forecast,
)


# =========================================================
# COMPANY CONFIGURATION
# =========================================================

COMPANY_NAME = "Mannstaedt GmbH"

COMPANY_LOGO = (
    PROJECT_ROOT
    / "app"
    / "assets"
    / "gmh_gruppe-logo.svg"
)

QR_QUERY = "source=qr"


# =========================================================
# COMPANY DPPs
# =========================================================

COMPANY_DPPS = [
    {
        "name": "Mannstaedt DPP",
        "path": (
            "/did:web:"
            "api.vera.spherity.dev:"
            "did-registry:"
            "gmh-gruppe-sn-00000006-c0a7c1a5"
        ),
        "real": True,
    },

    {
        "name": "Demo DPP 2",
        "path": (
            "/did:web:"
            "api.vera.spherity.dev:"
            "did-registry:"
            "compare-ec-ev-carbon-direct-20260820070207"
        ),
        "real": False,
    },

    {
        "name": "Demo DPP 3",
        "path": (
            "/did:web:"
            "api.vera.spherity.dev:"
            "did-registry:"
            "the-sustainable-white-1786975698821"
        ),
        "real": False,
    },
]


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title=f"{COMPANY_NAME} DPP Analytics",
    layout="wide",
)


# =========================================================
# CACHED API CALLS
# =========================================================

@st.cache_data(ttl=60, show_spinner=False)
def load_dpp_stats(
    start_at,
    end_at,
    path,
    query=None,
):
    return get_stats(
        start_at,
        end_at,
        path=path,
        query=query,
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_dpp_pageviews(
    start_at,
    end_at,
    path,
    query=None,
):
    return get_pageviews(
        start_at,
        end_at,
        unit="day",
        path=path,
        query=query,
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_dpp_metrics(
    start_at,
    end_at,
    metric_type,
    path,
    query=None,
):
    return get_metrics(
        start_at,
        end_at,
        metric_type,
        limit=50,
        path=path,
        query=query,
    )


# =========================================================
# HELPERS
# =========================================================

def datetime_to_ms(value):
    return int(value.timestamp() * 1000)


def seconds_to_readable(seconds):
    seconds = int(max(0, seconds))

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return f"{minutes}m {remaining_seconds}s"


def calculate_average_duration(
    total_time,
    visits,
):
    if visits <= 0:
        return 0

    return total_time / visits


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    if COMPANY_LOGO.exists():
        st.image(
            str(COMPANY_LOGO),
            width="stretch",
        )

    st.title("DPP Analytics")

    st.caption(COMPANY_NAME)

    st.divider()

    period = st.selectbox(
        "Date range",
        [
            "Last 7 days",
            "Last 30 days",
            "Last 12 months",
            "Custom",
        ],
        index=2,
    )

    now = datetime.now(timezone.utc)

    if period == "Last 7 days":

        start_date = now - timedelta(days=7)
        end_date = now

    elif period == "Last 30 days":

        start_date = now - timedelta(days=30)
        end_date = now

    elif period == "Last 12 months":

        start_date = now - timedelta(days=365)
        end_date = now

    else:

        dates = st.date_input(
            "Select date range",
            value=(
                (now - timedelta(days=365)).date(),
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

            start_date = now - timedelta(days=365)
            end_date = now

    st.divider()

    if st.button(
        "Refresh Data",
        width="stretch",
    ):
        st.cache_data.clear()
        st.rerun()


# =========================================================
# DATE CONVERSION
# =========================================================

start_at = datetime_to_ms(start_date)
end_at = datetime_to_ms(end_date)


# =========================================================
# HEADER
# =========================================================

logo_col, title_col = st.columns([1, 4])

with logo_col:

    if COMPANY_LOGO.exists():
        st.image(
            str(COMPANY_LOGO),
            width="stretch",
        )


with title_col:

    st.title(
        f"{COMPANY_NAME} DPP Analytics"
    )

    st.caption(
        "Digital Product Passport usage, QR analytics, "
        "audience insights and traffic prediction powered by VERA"
    )


st.caption(
    "Last refreshed: "
    + datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)

st.divider()


# =========================================================
# LOAD COMPANY DATA
# =========================================================

dpp_results = []

normal_timeseries = []

qr_timeseries = []

device_metrics_all = []

device_metrics_qr = []

region_metrics_all = []

region_metrics_qr = []


try:

    for dpp in COMPANY_DPPS:

        # -------------------------------------------------
        # ALL DPP TRAFFIC
        # -------------------------------------------------

        stats = load_dpp_stats(
            start_at,
            end_at,
            dpp["path"],
        )

        # -------------------------------------------------
        # QR TRAFFIC
        # -------------------------------------------------

        qr_stats = load_dpp_stats(
            start_at,
            end_at,
            dpp["path"],
            query=QR_QUERY,
        )

        views = stats.get(
            "pageviews",
            0,
        )

        qr_views = qr_stats.get(
            "pageviews",
            0,
        )

        visits = stats.get(
            "visits",
            0,
        )

        qr_visits = qr_stats.get(
            "visits",
            0,
        )

        total_time = stats.get(
            "totaltime",
            0,
        )

        qr_total_time = qr_stats.get(
            "totaltime",
            0,
        )

        avg_duration = calculate_average_duration(
            total_time,
            visits,
        )

        qr_avg_duration = calculate_average_duration(
            qr_total_time,
            qr_visits,
        )

        non_qr_visits = max(
            0,
            visits - qr_visits,
        )

        non_qr_total_time = max(
            0,
            total_time - qr_total_time,
        )

        non_qr_avg_duration = calculate_average_duration(
            non_qr_total_time,
            non_qr_visits,
        )

        dpp_results.append(
            {
                "name": dpp["name"],
                "path": dpp["path"],
                "views": views,
                "qr_views": qr_views,
                "qr_share": calculate_qr_share(
                    views,
                    qr_views,
                ),
                "visits": visits,
                "qr_visits": qr_visits,
                "total_time": total_time,
                "qr_total_time": qr_total_time,
                "avg_duration": avg_duration,
                "qr_avg_duration": qr_avg_duration,
                "non_qr_avg_duration": non_qr_avg_duration,
                "real": dpp["real"],
            }
        )

        # -------------------------------------------------
        # DAILY TRAFFIC
        # -------------------------------------------------

        normal_timeseries.append(
            load_dpp_pageviews(
                start_at,
                end_at,
                dpp["path"],
            )
        )

        # -------------------------------------------------
        # DAILY QR TRAFFIC
        # -------------------------------------------------

        qr_timeseries.append(
            load_dpp_pageviews(
                start_at,
                end_at,
                dpp["path"],
                query=QR_QUERY,
            )
        )

        # -------------------------------------------------
        # DEVICE ANALYTICS
        # -------------------------------------------------

        device_metrics_all.append(
            load_dpp_metrics(
                start_at,
                end_at,
                "device",
                dpp["path"],
            )
        )

        device_metrics_qr.append(
            load_dpp_metrics(
                start_at,
                end_at,
                "device",
                dpp["path"],
                query=QR_QUERY,
            )
        )

        # -------------------------------------------------
        # REGION ANALYTICS
        # -------------------------------------------------

        region_metrics_all.append(
            load_dpp_metrics(
                start_at,
                end_at,
                "region",
                dpp["path"],
            )
        )

        region_metrics_qr.append(
            load_dpp_metrics(
                start_at,
                end_at,
                "region",
                dpp["path"],
                query=QR_QUERY,
            )
        )


except Exception as e:

    st.error(
        f"Could not load Umami data: {e}"
    )

    st.stop()


# =========================================================
# AGGREGATE COMPANY ANALYTICS
# =========================================================

company = aggregate_dpp_stats(
    dpp_results
)

company_devices = aggregate_metric_lists(
    device_metrics_all
)

company_qr_devices = aggregate_metric_lists(
    device_metrics_qr
)

company_regions = aggregate_metric_lists(
    region_metrics_all
)

company_qr_regions = aggregate_metric_lists(
    region_metrics_qr
)


# =========================================================
# COMPANY-LEVEL DURATION VALUES
# =========================================================

company_total_visits = sum(
    item["visits"]
    for item in dpp_results
)

company_qr_visits = sum(
    item["qr_visits"]
    for item in dpp_results
)

company_total_time = sum(
    item["total_time"]
    for item in dpp_results
)

company_qr_total_time = sum(
    item["qr_total_time"]
    for item in dpp_results
)

company_non_qr_visits = max(
    0,
    company_total_visits
    - company_qr_visits,
)

company_non_qr_total_time = max(
    0,
    company_total_time
    - company_qr_total_time,
)

company_qr_avg_duration = calculate_average_duration(
    company_qr_total_time,
    company_qr_visits,
)

company_non_qr_avg_duration = calculate_average_duration(
    company_non_qr_total_time,
    company_non_qr_visits,
)


# =========================================================
# COMPANY OVERVIEW
# =========================================================

st.subheader(
    "Company Overview"
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Tracked DPPs",
    company["tracked_dpps"],
)

k2.metric(
    "DPP Pageviews",
    company["total_views"],
)

k3.metric(
    "QR Pageviews",
    company["qr_views"],
)

k4.metric(
    "QR Traffic Share",
    f"{company['qr_share_percent']}%",
)

st.divider()


# =========================================================
# QR VS NON-QR TRAFFIC
# =========================================================

st.subheader(
    "QR vs Non-QR Traffic"
)

qr_views = company["qr_views"]

total_views = company["total_views"]

non_qr_views = (
    calculate_non_qr_pageviews(
        total_views,
        qr_views,
    )
)

comparison = pd.DataFrame(
    {
        "Traffic Source": [
            "QR",
            "Non-QR",
        ],
        "Pageviews": [
            qr_views,
            non_qr_views,
        ],
    }
)

q1, q2, q3 = st.columns(3)

q1.metric(
    "QR Pageviews",
    qr_views,
)

q2.metric(
    "Non-QR Pageviews",
    non_qr_views,
)

q3.metric(
    "QR Traffic Share",
    f"{company['qr_share_percent']}%",
)

st.bar_chart(
    comparison.set_index(
        "Traffic Source"
    ),
    width="stretch",
)

st.divider()


# =========================================================
# QR AVERAGE DURATION ANALYTICS
# =========================================================

st.subheader(
    "QR Engagement — Average Visit Duration"
)

st.caption(
    "This section compares how long QR visitors stay "
    "with visitors arriving from other sources."
)

d1, d2, d3 = st.columns(3)

d1.metric(
    "QR Avg Visit Duration",
    seconds_to_readable(
        company_qr_avg_duration
    ),
)

d2.metric(
    "Non-QR Avg Visit Duration",
    seconds_to_readable(
        company_non_qr_avg_duration
    ),
)

duration_difference = (
    company_qr_avg_duration
    - company_non_qr_avg_duration
)

d3.metric(
    "QR Duration Difference",
    seconds_to_readable(
        abs(duration_difference)
    ),
    delta=(
        "QR longer"
        if duration_difference > 0
        else "QR shorter"
        if duration_difference < 0
        else "Same"
    ),
)


duration_comparison = pd.DataFrame(
    {
        "Traffic Source": [
            "QR",
            "Non-QR",
        ],
        "Avg Duration (seconds)": [
            round(
                company_qr_avg_duration,
                2,
            ),
            round(
                company_non_qr_avg_duration,
                2,
            ),
        ],
    }
)


st.bar_chart(
    duration_comparison.set_index(
        "Traffic Source"
    ),
    width="stretch",
)



# =========================================================
# QR DURATION BY DPP
# =========================================================

st.subheader(
    "Average Visit Duration by DPP"
)


duration_df = pd.DataFrame(
    [
        {
            "DPP": item["name"],
            "QR Avg Duration":
                seconds_to_readable(
                    item["qr_avg_duration"]
                ),
            "Non-QR Avg Duration":
                seconds_to_readable(
                    item["non_qr_avg_duration"]
                ),
            "QR Avg Seconds":
                round(
                    item["qr_avg_duration"],
                    2,
                ),
            "Non-QR Avg Seconds":
                round(
                    item["non_qr_avg_duration"],
                    2,
                ),
        }
        for item in dpp_results
    ]
)


st.dataframe(
    duration_df[
        [
            "DPP",
            "QR Avg Duration",
            "Non-QR Avg Duration",
        ]
    ],
    width="stretch",
    hide_index=True,
)


duration_chart_df = duration_df[
    [
        "DPP",
        "QR Avg Seconds",
        "Non-QR Avg Seconds",
    ]
].set_index("DPP")


st.bar_chart(
    duration_chart_df,
    width="stretch",
)


st.divider()


# =========================================================
# TRAFFIC OVER TIME
# =========================================================

st.subheader(
    "DPP Traffic Over Time"
)

company_traffic = aggregate_timeseries(
    normal_timeseries
)

company_qr_traffic = aggregate_timeseries(
    qr_timeseries
)

if not company_traffic.empty:

    chart = company_traffic.rename(
        columns={
            "views": "All DPP Views"
        }
    )

    if not company_qr_traffic.empty:

        qr_chart = (
            company_qr_traffic.rename(
                columns={
                    "views": "QR Views"
                }
            )
        )

        chart = chart.merge(
            qr_chart,
            on="date",
            how="outer",
        )

    chart = (
        chart
        .fillna(0)
        .sort_values("date")
        .set_index("date")
    )

    st.line_chart(
        chart,
        width="stretch",
    )

else:

    st.info(
        "No traffic data found."
    )

st.divider()


# =========================================================
# DEVICE TYPE ANALYTICS
# =========================================================

st.subheader(
    "Device Type Analytics"
)

device_col1, device_col2 = st.columns(2)

with device_col1:

    st.markdown(
        "**All DPP Traffic**"
    )

    if company_devices:

        device_df = (
            pd.DataFrame(
                company_devices
            )
            .rename(
                columns={
                    "x": "Device",
                    "y": "Visitors",
                }
            )
        )

        st.bar_chart(
            device_df.set_index(
                "Device"
            ),
            width="stretch",
        )

        st.dataframe(
            device_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No device data available."
        )


with device_col2:

    st.markdown(
        "**QR Traffic**"
    )

    if company_qr_devices:

        qr_device_df = (
            pd.DataFrame(
                company_qr_devices
            )
            .rename(
                columns={
                    "x": "Device",
                    "y": "QR Visitors",
                }
            )
        )

        st.bar_chart(
            qr_device_df.set_index(
                "Device"
            ),
            width="stretch",
        )

        st.dataframe(
            qr_device_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No QR device data available."
        )


st.divider()


# =========================================================
# REGION ANALYTICS
# =========================================================

st.subheader(
    "Region Analytics"
)

region_col1, region_col2 = st.columns(2)

with region_col1:

    st.markdown(
        "**Top Regions — All Traffic**"
    )

    if company_regions:

        region_df = (
            pd.DataFrame(
                company_regions[:10]
            )
            .rename(
                columns={
                    "x": "Region",
                    "y": "Visitors",
                }
            )
        )

        st.bar_chart(
            region_df.set_index(
                "Region"
            ),
            width="stretch",
        )

        st.dataframe(
            region_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No region data available."
        )


with region_col2:

    st.markdown(
        "**Top Regions — QR Traffic**"
    )

    if company_qr_regions:

        qr_region_df = (
            pd.DataFrame(
                company_qr_regions[:10]
            )
            .rename(
                columns={
                    "x": "Region",
                    "y": "QR Visitors",
                }
            )
        )

        st.bar_chart(
            qr_region_df.set_index(
                "Region"
            ),
            width="stretch",
        )

        st.dataframe(
            qr_region_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No QR region data available."
        )


st.divider()


# =========================================================
# DPP PERFORMANCE
# =========================================================

st.subheader(
    "DPP Performance"
)

comparison_df = pd.DataFrame(
    dpp_results
)

comparison_df = (
    comparison_df[
        [
            "name",
            "views",
        ]
    ]
    .rename(
        columns={
            "name": "DPP",
            "views": "Views",
        }
    )
)

if not comparison_df.empty:

    st.bar_chart(
        comparison_df.set_index(
            "DPP"
        ),
        width="stretch",
    )


st.divider()


# =========================================================
# MACHINE LEARNING
# =========================================================

st.header(
    "Machine Learning Traffic Forecast"
)

st.caption(
    "Experimental 7-day forecast based on historical "
    "company DPP pageviews."
)



ml_result = build_company_forecast(
    company_traffic,
    days=7,
)


with st.expander(
    "1. Raw Traffic Data"
):

    st.dataframe(
        company_traffic,
        width="stretch",
        hide_index=True,
    )


with st.expander(
    "2. Cleaned Daily Data"
):

    cleaned = ml_result[
        "cleaned"
    ]

    st.dataframe(
        cleaned,
        width="stretch",
        hide_index=True,
    )


with st.expander(
    "3. ML Features"
):

    features = ml_result[
        "features"
    ]

    if not features.empty:

        st.dataframe(
            features,
            width="stretch",
            hide_index=True,
        )


with st.expander(
    "4. Normalized ML Input"
):

    normalized = ml_result[
        "normalized"
    ]

    if not normalized.empty:

        st.dataframe(
            normalized,
            width="stretch",
            hide_index=True,
        )


if ml_result["status"] == "not_enough_data":

    st.warning(
        "Not enough historical traffic data "
        "to train the ML model yet."
    )

else:

    forecast = ml_result[
        "forecast"
    ]

    predicted_total = int(
        forecast[
            "predicted_pageviews"
        ].sum()
    )

    predicted_average = round(
        forecast[
            "predicted_pageviews"
        ].mean(),
        1,
    )

    mae = ml_result[
        "mae"
    ]

    ml1, ml2, ml3 = st.columns(3)

    ml1.metric(
        "Predicted Next 7 Days",
        predicted_total,
    )

    ml2.metric(
        "Predicted Daily Average",
        predicted_average,
    )

    ml3.metric(
        "Validation MAE",
        mae,
    )


    historical = (
        ml_result[
            "cleaned"
        ][
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
        forecast[
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

    ml_chart = (
        historical.merge(
            prediction_chart,
            on="date",
            how="outer",
        )
        .sort_values("date")
        .set_index("date")
    )

    st.subheader(
        "Actual Traffic and Forecast"
    )

    st.line_chart(
        ml_chart,
        width="stretch",
    )


st.divider()


# =========================================================
# TOP DPP
# =========================================================

if company["top_dpp"]:

    top = company[
        "top_dpp"
    ]

    st.subheader(
        "Top Performing DPP"
    )

    st.success(
        f"**{top['name']}** currently has "
        f"the highest traffic with "
        f"**{top['views']} views**."
    )


# =========================================================
# DEMO NOTE
# =========================================================

st.divider()

st.caption(
    """
Prototype note: the Mannstaedt DPP is the real company DPP.
The two additional DPPs are temporary placeholders used only
to provide more traffic data for this prototype.
"""
)
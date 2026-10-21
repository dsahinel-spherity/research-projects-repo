from __future__ import annotations

import pandas as pd


SECTION_EVENT = (
    "dpp-section-view"
)

DOCUMENT_EVENT = (
    "dpp-document-open"
)


# =========================================================
# GENERAL
# =========================================================

def calculate_qr_share(
    total_views,
    qr_views,
):

    if (
        total_views
        == 0
    ):

        return 0.0

    return round(
        qr_views
        / total_views
        * 100,
        2,
    )


# =========================================================
# GMH COMPANY KPIS
# =========================================================

def real_company_kpis(
    stats,
    events_df,
    sessions_df,
):

    if (
        events_df is None
        or events_df.empty
    ):

        return {
            "pageviews": 0,
            "visitors": 0,
            "visits": 0,
            "average_visit_seconds": 0.0,
            "section_views": 0,
            "document_opens": 0,
            "session_records": 0,
        }

    events = (
        events_df.copy()
    )

    if (
        "eventType"
        in events.columns
    ):

        pageviews = int(
            (
                events[
                    "eventType"
                ]
                == 1
            )
            .sum()
        )

    else:

        if (
            "eventName"
            in events.columns
        ):

            pageviews = int(
                events[
                    "eventName"
                ]
                .isna()
                .sum()
            )

        else:

            pageviews = 0

    visits = (
        events[
            "sessionId"
        ]
        .nunique()
        if (
            "sessionId"
            in events.columns
        )
        else 0
    )

    visitors = (
        visits
    )

    section_views = 0

    document_opens = 0

    if (
        "eventName"
        in events.columns
    ):

        section_views = int(
            (
                events[
                    "eventName"
                ]
                == SECTION_EVENT
            )
            .sum()
        )

        document_opens = int(
            (
                events[
                    "eventName"
                ]
                == DOCUMENT_EVENT
            )
            .sum()
        )

    average_visit_seconds = 0.0

    if (
        "sessionId"
        in events.columns
        and "createdAt"
        in events.columns
    ):

        duration_data = (
            events[
                [
                    "sessionId",
                    "createdAt",
                ]
            ]
            .copy()
        )

        duration_data[
            "createdAt"
        ] = (
            pd.to_datetime(
                duration_data[
                    "createdAt"
                ],
                errors="coerce",
                utc=True,
            )
        )

        duration_data = (
            duration_data
            .dropna(
                subset=[
                    "sessionId",
                    "createdAt",
                ]
            )
        )

        if (
            not duration_data.empty
        ):

            duration_df = (
                duration_data
                .groupby(
                    "sessionId"
                )[
                    "createdAt"
                ]
                .agg(
                    [
                        "min",
                        "max",
                    ]
                )
            )

            duration_df[
                "duration"
            ] = (
                duration_df[
                    "max"
                ]
                - duration_df[
                    "min"
                ]
            ).dt.total_seconds()

            average_visit_seconds = (
                float(
                    duration_df[
                        "duration"
                    ]
                    .mean()
                )
            )

    return {
        "pageviews":
            pageviews,

        "visitors":
            visitors,

        "visits":
            visits,

        "average_visit_seconds":
            round(
                average_visit_seconds,
                1,
            ),

        "section_views":
            section_views,

        "document_opens":
            document_opens,

        "session_records":
            len(
                sessions_df
            ),
    }


# =========================================================
# SECTION PERFORMANCE
# =========================================================

def section_performance(
    events_df,
):

    if (
        events_df is None
        or events_df.empty
        or "eventName"
        not in events_df.columns
    ):

        return (
            pd.DataFrame()
        )

    sections = (
        events_df[
            events_df[
                "eventName"
            ]
            == SECTION_EVENT
        ]
        .copy()
    )

    if (
        sections.empty
        or "section"
        not in sections.columns
    ):

        return (
            pd.DataFrame()
        )

    sections = (
        sections[
            sections[
                "section"
            ]
            .notna()
        ]
    )

    if sections.empty:

        return (
            pd.DataFrame()
        )

    if (
        "sessionId"
        in sections.columns
    ):

        result = (
            sections
            .groupby(
                "section",
                as_index=False,
            )
            .agg(
                views=(
                    "section",
                    "size",
                ),

                unique_sessions=(
                    "sessionId",
                    "nunique",
                ),
            )
        )

    else:

        result = (
            sections
            .groupby(
                "section",
                as_index=False,
            )
            .agg(
                views=(
                    "section",
                    "size",
                ),
            )
        )

        result[
            "unique_sessions"
        ] = (
            result[
                "views"
            ]
        )

    total_views = (
        result[
            "views"
        ]
        .sum()
    )

    result[
        "view_share_percent"
    ] = (
        result[
            "views"
        ]
        / total_views
        * 100
        if total_views
        else 0
    )

    return (
        result
        .sort_values(
            "views",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# DOCUMENT PERFORMANCE
# =========================================================

def document_performance(
    events_df,
):

    if (
        events_df is None
        or events_df.empty
        or "eventName"
        not in events_df.columns
    ):

        return (
            pd.DataFrame()
        )

    documents = (
        events_df[
            events_df[
                "eventName"
            ]
            == DOCUMENT_EVENT
        ]
        .copy()
    )

    if (
        documents.empty
        or "document"
        not in documents.columns
    ):

        return (
            pd.DataFrame()
        )

    documents = (
        documents[
            documents[
                "document"
            ]
            .notna()
        ]
    )

    if documents.empty:

        return (
            pd.DataFrame()
        )

    if (
        "sessionId"
        in documents.columns
    ):

        result = (
            documents
            .groupby(
                "document",
                as_index=False,
            )
            .agg(
                opens=(
                    "document",
                    "size",
                ),

                unique_sessions=(
                    "sessionId",
                    "nunique",
                ),
            )
        )

    else:

        result = (
            documents
            .groupby(
                "document",
                as_index=False,
            )
            .agg(
                opens=(
                    "document",
                    "size",
                ),
            )
        )

        result[
            "unique_sessions"
        ] = (
            result[
                "opens"
            ]
        )

    if (
        "source"
        in documents.columns
    ):

        qr_counts = (
            documents[
                documents[
                    "source"
                ]
                .astype(str)
                .str.lower()
                == "qr"
            ]
            .groupby(
                "document"
            )
            .size()
        )

        result[
            "qr_opens"
        ] = (
            result[
                "document"
            ]
            .map(
                qr_counts
            )
            .fillna(0)
            .astype(int)
        )

        result[
            "qr_share_percent"
        ] = (
            result[
                "qr_opens"
            ]
            / result[
                "opens"
            ]
            * 100
        )

    return (
        result
        .sort_values(
            "opens",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# DPP PERFORMANCE
# =========================================================

def dpp_performance(
    events_df,
):

    if (
        events_df is None
        or events_df.empty
        or "dpp_id"
        not in events_df.columns
    ):

        return (
            pd.DataFrame()
        )

    events = (
        events_df[
            events_df[
                "dpp_id"
            ]
            .notna()
        ]
        .copy()
    )

    events = (
        events[
            events[
                "dpp_id"
            ]
            .astype(str)
            != ""
        ]
    )

    if events.empty:

        return (
            pd.DataFrame()
        )

    rows = []

    for dpp_id, group in (
        events.groupby(
            "dpp_id"
        )
    ):

        section_views = int(
            (
                group[
                    "eventName"
                ]
                == SECTION_EVENT
            )
            .sum()
        )

        document_opens = int(
            (
                group[
                    "eventName"
                ]
                == DOCUMENT_EVENT
            )
            .sum()
        )

        sessions = (
            group[
                "sessionId"
            ]
            .nunique()
            if (
                "sessionId"
                in group.columns
            )
            else 0
        )

        rows.append(
            {
                "dpp_id":
                    dpp_id,

                "sessions":
                    sessions,

                "section_views":
                    section_views,

                "document_opens":
                    document_opens,

                "engagement_events":
                    (
                        section_views
                        + document_opens
                    ),

                "documents_per_session":
                    (
                        document_opens
                        / sessions
                        if sessions
                        else 0
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "engagement_events",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# UMAMI METRICS
# =========================================================

def metrics_to_dataframe(
    metrics,
    name_column,
):

    if metrics is None:

        return (
            pd.DataFrame(
                columns=[
                    name_column,
                    "count",
                ]
            )
        )

    if isinstance(
        metrics,
        pd.DataFrame,
    ):

        df = (
            metrics.copy()
        )

        if df.empty:

            return (
                pd.DataFrame(
                    columns=[
                        name_column,
                        "count",
                    ]
                )
            )

    else:

        if (
            len(
                metrics
            )
            == 0
        ):

            return (
                pd.DataFrame(
                    columns=[
                        name_column,
                        "count",
                    ]
                )
            )

        df = (
            pd.DataFrame(
                metrics
            )
        )

    if (
        "x"
        not in df.columns
        or "y"
        not in df.columns
    ):

        return (
            pd.DataFrame(
                columns=[
                    name_column,
                    "count",
                ]
            )
        )

    result = (
        df.rename(
            columns={
                "x":
                    name_column,

                "y":
                    "count",
            }
        )[
            [
                name_column,
                "count",
            ]
        ]
        .copy()
    )

    result[
        "count"
    ] = (
        pd.to_numeric(
            result[
                "count"
            ],
            errors="coerce",
        )
        .fillna(0)
    )

    return (
        result
        .sort_values(
            "count",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# PAGEVIEW TIMESERIES
# =========================================================

def normalize_pageview_timeseries(
    pageviews_df,
):

    if (
        pageviews_df is None
        or pageviews_df.empty
    ):

        return (
            pd.DataFrame(
                columns=[
                    "date",
                    "pageviews",
                ]
            )
        )

    result = (
        pageviews_df.copy()
    )

    if (
        "date"
        not in result.columns
        and "x"
        in result.columns
    ):

        result[
            "date"
        ] = (
            pd.to_datetime(
                result[
                    "x"
                ],
                errors="coerce",
                utc=True,
            )
        )

    if (
        "pageviews"
        not in result.columns
        and "y"
        in result.columns
    ):

        result[
            "pageviews"
        ] = (
            pd.to_numeric(
                result[
                    "y"
                ],
                errors="coerce",
            )
            .fillna(0)
        )

    if (
        "date"
        not in result.columns
        or "pageviews"
        not in result.columns
    ):

        return (
            pd.DataFrame(
                columns=[
                    "date",
                    "pageviews",
                ]
            )
        )

    return (
        result[
            [
                "date",
                "pageviews",
            ]
        ]
        .dropna(
            subset=[
                "date",
            ]
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )
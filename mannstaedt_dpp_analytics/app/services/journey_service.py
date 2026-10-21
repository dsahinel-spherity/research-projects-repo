from __future__ import annotations

from urllib.parse import parse_qs, urlsplit, urlunsplit

import pandas as pd

from app.config import DPP_VIEWER_BASE_URL


SECTION_EVENT = "dpp-section-view"
DOCUMENT_EVENT = "dpp-document-open"

JOURNEY_INACTIVITY_MINUTES = 10


def recorded_dpp_location(custom, dpp_id, metadata):
    """Use the path recorded for this DPP, never infer a route from its ID."""
    matching = custom[custom['dpp_id'].fillna('').astype(str).eq(dpp_id)]
    for _, event in matching.iterrows():
        path = _safe_string(event.get('url_path'))
        if not path:
            continue
        parsed = urlsplit(path)
        if parsed.scheme in {'http', 'https'} and parsed.hostname and not parsed.username:
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', '')), parsed.path
        if parsed.scheme or parsed.netloc or not path.startswith('/'):
            continue
        host = (_safe_string(event.get('hostname'))
                or _safe_string(metadata.get('hostname'))
                or DPP_VIEWER_BASE_URL)
        if host:
            origin = urlsplit(host if '://' in host else 'https://' + host)
            if origin.scheme in {'http', 'https'} and origin.hostname and not origin.username and origin.path in {'', '/'}:
                return urlunsplit((origin.scheme, origin.netloc, parsed.path, '', '')), parsed.path
        return '', parsed.path
    return '', ''


# =========================================================
# HELPERS
# =========================================================

def _safe_string(
    value,
):

    if value is None:

        return ""

    try:

        if pd.isna(
            value
        ):

            return ""

    except Exception:

        pass

    return (
        str(
            value
        )
        .strip()
    )


def _source_from_query(
    query,
):

    query = (
        _safe_string(
            query
        )
    )

    if not query:

        return "web"

    parsed = (
        parse_qs(
            query.lstrip(
                "?"
            )
        )
    )

    value = (
        parsed.get(
            "source",
            [""],
        )[0]
        .lower()
    )

    if (
        value
        == "qr"
    ):

        return "qr"

    return "web"


def _collapse_consecutive(
    values,
):

    result = []

    for value in values:

        if (
            not result
            or result[
                -1
            ]
            != value
        ):

            result.append(
                value
            )

    return result


# =========================================================
# EVENT STREAM
# =========================================================

def prepare_event_stream(
    events_df,
):

    if (
        events_df is None
        or events_df.empty
    ):

        return (
            pd.DataFrame()
        )

    events = (
        events_df.copy()
    )

    rename_map = {
        "id":
            "event_id",

        "sessionId":
            "session_id",

        "createdAt":
            "timestamp",

        "eventName":
            "event_name",

        "urlPath":
            "url_path",

        "urlQuery":
            "url_query",

        "pageTitle":
            "page_title",
    }

    events = (
        events.rename(
            columns={
                key:
                    value

                for key, value
                in rename_map.items()

                if key
                in events.columns
            }
        )
    )

    defaults = {
        "event_id":
            "",

        "session_id":
            "",

        "timestamp":
            pd.NaT,

        "event_name":
            "",

        "url_path":
            "",

        "url_query":
            "",

        "page_title":
            "",

        "section":
            "",

        "document":
            "",

        "dpp_id":
            "",

        "organization":
            "",

        "source":
            "",
    }

    for column, default in (
        defaults.items()
    ):

        if (
            column
            not in events.columns
        ):

            events[
                column
            ] = default

    events[
        "timestamp"
    ] = (
        pd.to_datetime(
            events[
                "timestamp"
            ],
            errors="coerce",
            utc=True,
        )
    )

    events[
        "event_type"
    ] = "pageview"

    events.loc[
        events[
            "event_name"
        ]
        == SECTION_EVENT,
        "event_type",
    ] = "section_view"

    events.loc[
        events[
            "event_name"
        ]
        == DOCUMENT_EVENT,
        "event_type",
    ] = "document_open"

    return (
        events
        .sort_values(
            [
                "session_id",
                "timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# SPLIT SESSION BY INACTIVITY
# =========================================================

def _split_session_into_journeys(
    group,
    inactivity_minutes=
        JOURNEY_INACTIVITY_MINUTES,
):

    if (
        group is None
        or group.empty
    ):

        return []

    group = (
        group
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    if '_journey_segment' in group.columns:
        return [chunk.copy() for _, chunk in group.groupby('_journey_segment', sort=False)]

    gap_limit = (
        pd.Timedelta(
            minutes=
                inactivity_minutes
        )
    )

    journeys = []

    start_index = 0

    for index in range(
        1,
        len(
            group
        ),
    ):

        previous_time = (
            group.loc[
                index - 1,
                "timestamp",
            ]
        )

        current_time = (
            group.loc[
                index,
                "timestamp",
            ]
        )

        if (
            pd.isna(
                previous_time
            )
            or pd.isna(
                current_time
            )
        ):

            continue

        gap = (
            current_time
            - previous_time
        )

        if (
            gap
            > gap_limit
        ):

            chunk = (
                group.iloc[
                    start_index:index
                ]
                .copy()
            )

            if (
                not chunk.empty
            ):

                journeys.append(
                    chunk
                )

            start_index = (
                index
            )

    final_chunk = (
        group.iloc[
            start_index:
        ]
        .copy()
    )

    if (
        not final_chunk.empty
    ):

        journeys.append(
            final_chunk
        )

    return journeys


# =========================================================
# BUILD JOURNEYS
# =========================================================

def build_user_journeys(
    events_df,
    sessions_df,
):

    events = (
        prepare_event_stream(
            events_df
        )
    )

    if events.empty:

        return (
            pd.DataFrame()
        )

    session_lookup = {}

    if (
        sessions_df is not None
        and not sessions_df.empty
    ):

        sessions = (
            sessions_df.copy()
        )

        if (
            "id"
            in sessions.columns
        ):

            sessions = (
                sessions.rename(
                    columns={
                        "id":
                            "session_id",
                    }
                )
            )

        for _, row in (
            sessions.iterrows()
        ):

            session_lookup[
                _safe_string(
                    row.get(
                        "session_id"
                    )
                )
            ] = (
                row.to_dict()
            )

    rows = []

    for (
        session_id,
        full_group,
    ) in events.groupby(
        "session_id",
        dropna=False,
    ):

        journey_groups = (
            _split_session_into_journeys(
                full_group,
                inactivity_minutes=
                    JOURNEY_INACTIVITY_MINUTES,
            )
        )

        for (
            journey_number,
            group,
        ) in enumerate(
            journey_groups,
            start=1,
        ):

            group = (
                group
                .sort_values(
                    "timestamp"
                )
                .reset_index(
                    drop=True
                )
            )

            custom = (
                group[
                    group[
                        "event_type"
                    ]
                    .isin(
                        [
                            "section_view",
                            "document_open",
                        ]
                    )
                ]
            )

            if custom.empty:

                continue

            journey_id = (
                f"{session_id}-"
                f"{journey_number}"
            )

            # -------------------------------------------------
            # ENTRY
            # -------------------------------------------------

            pageviews = (
                group[
                    group[
                        "event_type"
                    ]
                    == "pageview"
                ]
            )

            if (
                not pageviews.empty
            ):

                first_pageview = (
                    pageviews.iloc[
                        0
                    ]
                )

                entry_path = (
                    _safe_string(
                        first_pageview[
                            "url_path"
                        ]
                    )
                )

                entry_query = (
                    _safe_string(
                        first_pageview[
                            "url_query"
                        ]
                    )
                )

                source = (
                    _source_from_query(
                        entry_query
                    )
                )

            else:

                entry_path = ""

                entry_query = ""

                source = "web"

            # -------------------------------------------------
            # CUSTOM SOURCE
            # -------------------------------------------------

            custom_sources = (
                custom[
                    "source"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
            )

            if (
                custom_sources
                .eq(
                    "qr"
                )
                .any()
            ):

                source = "qr"

            # -------------------------------------------------
            # SECTIONS
            # -------------------------------------------------

            sections = (
                custom[
                    custom[
                        "event_type"
                    ]
                    == "section_view"
                ][
                    "section"
                ]
                .fillna("")
                .astype(str)
                .tolist()
            )

            sections = [
                section
                for section
                in sections
                if section
            ]

            sections = (
                _collapse_consecutive(
                    sections
                )
            )

            # -------------------------------------------------
            # DOCUMENTS
            # -------------------------------------------------

            documents = (
                custom[
                    custom[
                        "event_type"
                    ]
                    == "document_open"
                ][
                    "document"
                ]
                .fillna("")
                .astype(str)
                .tolist()
            )

            documents = [
                document
                for document
                in documents
                if document
            ]

            # -------------------------------------------------
            # DPP
            # -------------------------------------------------

            dpp_values = (
                custom[
                    "dpp_id"
                ]
                .fillna("")
                .astype(str)
                .tolist()
            )

            dpp_values = [
                value
                for value
                in dpp_values
                if value
            ]

            dpp_id = (
                dpp_values[
                    0
                ]
                if dpp_values
                else entry_path
            )

            # -------------------------------------------------
            # ORGANIZATION
            # -------------------------------------------------

            organization_values = (
                custom[
                    "organization"
                ]
                .fillna("")
                .astype(str)
                .tolist()
            )

            organization_values = [
                value
                for value
                in organization_values
                if value
            ]

            organization = (
                organization_values[
                    0
                ]
                if organization_values
                else ""
            )

            # -------------------------------------------------
            # JOURNEY PATH
            # -------------------------------------------------

            steps = [
                f"Entry ({source.upper()})"
            ]

            for _, event in (
                custom.iterrows()
            ):

                if (
                    event[
                        "event_type"
                    ]
                    == "section_view"
                ):

                    section = (
                        _safe_string(
                            event[
                                "section"
                            ]
                        )
                    )

                    if section:

                        label = (
                            f"Section: {section}"
                        )

                        if (
                            steps[
                                -1
                            ]
                            != label
                        ):

                            steps.append(
                                label
                            )

                elif (
                    event[
                        "event_type"
                    ]
                    == "document_open"
                ):

                    document = (
                        _safe_string(
                            event[
                                "document"
                            ]
                        )
                    )

                    if document:

                        steps.append(
                            f"Document: {document}"
                        )

            # -------------------------------------------------
            # DURATION
            # -------------------------------------------------

            first_time = (
                group[
                    "timestamp"
                ]
                .min()
            )

            last_time = (
                group[
                    "timestamp"
                ]
                .max()
            )

            duration = 0.0

            if (
                pd.notna(
                    first_time
                )
                and pd.notna(
                    last_time
                )
            ):

                duration = max(
                    0.0,
                    (
                        last_time
                        - first_time
                    )
                    .total_seconds(),
                )

            metadata = (
                session_lookup.get(
                    _safe_string(
                        session_id
                    ),
                    {},
                )
            )

            dpp_url, dpp_path = recorded_dpp_location(custom, dpp_id, metadata)

            rows.append(
                {
                    "journey_id":
                        journey_id,

                    "session_id":
                        session_id,

                    "journey_number":
                        journey_number,

                    "source":
                        source,

                    "dpp_id":
                        dpp_id,

                    "dpp_url": dpp_url,
                    "dpp_path": dpp_path,
                    "dpp_count": len(set(dpp_values)),

                    "organization":
                        organization,

                    "entry_path":
                        entry_path,

                    "entry_query":
                        entry_query,

                    "first_section":
                        (
                            sections[
                                0
                            ]
                            if sections
                            else ""
                        ),

                    "last_section":
                        (
                            sections[
                                -1
                            ]
                            if sections
                            else ""
                        ),

                    "sections":
                        sections,

                    "section_count":
                        len(
                            sections
                        ),

                    "unique_section_count":
                        len(
                            set(
                                sections
                            )
                        ),

                    "documents":
                        documents,

                    "document_open_count":
                        len(
                            documents
                        ),

                    "opened_document":
                        bool(
                            documents
                        ),

                    "journey_steps":
                        steps,

                    "journey_path":
                        " → ".join(
                            steps
                        ),

                    "started_at":
                        first_time,

                    "ended_at":
                        last_time,

                    "duration_seconds":
                        duration,

                    "device":
                        metadata.get(
                            "device",
                            "",
                        ),

                    "browser":
                        metadata.get(
                            "browser",
                            "",
                        ),

                    "os":
                        metadata.get(
                            "os",
                            "",
                        ),

                    "country":
                        metadata.get(
                            "country",
                            "",
                        ),

                    "region":
                        (
                            metadata.get(
                                "subdivision1",
                                "",
                            )
                            or metadata.get(
                                "region",
                                "",
                            )
                        ),

                    "city":
                        metadata.get(
                            "city",
                            "",
                        ),
                }
            )

    return (
        pd.DataFrame(
            rows
        )
    )


# =========================================================
# KPIS
# =========================================================

def journey_kpis(
    journeys_df,
):

    if (
        journeys_df is None
        or journeys_df.empty
    ):

        return {
            "sessions": 0,
            "qr_sessions": 0,
            "web_sessions": 0,
            "document_sessions": 0,
            "document_conversion_percent": 0.0,
            "qr_document_conversion_percent": 0.0,
            "web_document_conversion_percent": 0.0,
            "avg_sections": 0.0,
            "avg_duration_seconds": 0.0,
        }

    journeys = (
        journeys_df.copy()
    )

    qr = (
        journeys[
            journeys[
                "source"
            ]
            == "qr"
        ]
    )

    web = (
        journeys[
            journeys[
                "source"
            ]
            != "qr"
        ]
    )

    def conversion(
        frame,
    ):

        if frame.empty:

            return 0.0

        return (
            frame[
                "opened_document"
            ]
            .astype(float)
            .mean()
            * 100
        )

    return {
        "sessions":
            len(
                journeys
            ),

        "qr_sessions":
            len(
                qr
            ),

        "web_sessions":
            len(
                web
            ),

        "document_sessions":
            int(
                journeys[
                    "opened_document"
                ]
                .astype(int)
                .sum()
            ),

        "document_conversion_percent":
            round(
                conversion(
                    journeys
                ),
                2,
            ),

        "qr_document_conversion_percent":
            round(
                conversion(
                    qr
                ),
                2,
            ),

        "web_document_conversion_percent":
            round(
                conversion(
                    web
                ),
                2,
            ),

        "avg_sections":
            round(
                float(
                    journeys[
                        "section_count"
                    ]
                    .mean()
                ),
                2,
            ),

        "avg_duration_seconds":
            round(
                float(
                    journeys[
                        "duration_seconds"
                    ]
                    .mean()
                ),
                1,
            ),
    }


# =========================================================
# TOP JOURNEYS
# =========================================================

def top_journeys(
    journeys_df,
    limit=15,
):

    if (
        journeys_df is None
        or journeys_df.empty
    ):

        return (
            pd.DataFrame()
        )

    return (
        journeys_df[
            "journey_path"
        ]
        .value_counts()
        .head(
            limit
        )
        .rename_axis(
            "journey"
        )
        .reset_index(
            name="journeys"
        )
    )


# =========================================================
# ENTRY / EXIT
# =========================================================

def entry_exit_sections(
    journeys_df,
):

    if (
        journeys_df is None
        or journeys_df.empty
    ):

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    entry = (
        journeys_df[
            "first_section"
        ]
        .replace(
            "",
            pd.NA,
        )
        .dropna()
        .value_counts()
        .rename_axis(
            "section"
        )
        .reset_index(
            name="journeys"
        )
    )

    exit_df = (
        journeys_df[
            "last_section"
        ]
        .replace(
            "",
            pd.NA,
        )
        .dropna()
        .value_counts()
        .rename_axis(
            "section"
        )
        .reset_index(
            name="journeys"
        )
    )

    return (
        entry,
        exit_df,
    )


# =========================================================
# SECTION -> DOCUMENT
# =========================================================

def section_document_conversion(
    events_df,
):

    events = (
        prepare_event_stream(
            events_df
        )
    )

    if events.empty:

        return (
            pd.DataFrame()
        )

    rows = []

    for (
        session_id,
        full_group,
    ) in events.groupby(
        "session_id",
        dropna=False,
    ):

        journey_groups = (
            _split_session_into_journeys(
                full_group
            )
        )

        for (
            journey_number,
            group,
        ) in enumerate(
            journey_groups,
            start=1,
        ):

            group = (
                group
                .sort_values(
                    "timestamp"
                )
                .reset_index(
                    drop=True
                )
            )

            journey_id = (
                f"{session_id}-"
                f"{journey_number}"
            )

            for index, event in (
                group.iterrows()
            ):

                if (
                    event[
                        "event_type"
                    ]
                    != "section_view"
                ):

                    continue

                section = (
                    _safe_string(
                        event[
                            "section"
                        ]
                    )
                )

                if not section:

                    continue

                later = (
                    group.iloc[
                        index + 1:
                    ]
                )

                document_after = (
                    later[
                        "event_type"
                    ]
                    .eq(
                        "document_open"
                    )
                    .any()
                )

                rows.append(
                    {
                        "journey_id":
                            journey_id,

                        "section":
                            section,

                        "document_after":
                            bool(
                                document_after
                            ),
                    }
                )

    if not rows:

        return (
            pd.DataFrame()
        )

    data = (
        pd.DataFrame(
            rows
        )
    )

    result = (
        data
        .groupby(
            "section",
            as_index=False,
        )
        .agg(
            section_views=(
                "section",
                "size",
            ),

            journeys=(
                "journey_id",
                "nunique",
            ),

            document_conversions=(
                "document_after",
                "sum",
            ),

            document_conversion_rate=(
                "document_after",
                "mean",
            ),
        )
    )

    result[
        "document_conversion_percent"
    ] = (
        result[
            "document_conversion_rate"
        ]
        * 100
    )

    return (
        result
        .sort_values(
            "document_conversion_percent",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# TRANSITIONS
# =========================================================

def journey_transitions(
    journeys_df,
):

    if (
        journeys_df is None
        or journeys_df.empty
    ):

        return (
            pd.DataFrame()
        )

    rows = []

    for _, journey in (
        journeys_df.iterrows()
    ):

        steps = (
            journey[
                "journey_steps"
            ]
        )

        if (
            not isinstance(
                steps,
                list,
            )
            or len(
                steps
            )
            < 2
        ):

            continue

        for index in range(
            len(
                steps
            )
            - 1
        ):

            rows.append(
                {
                    "from":
                        steps[
                            index
                        ],

                    "to":
                        steps[
                            index + 1
                        ],

                    "journey_id":
                        journey[
                            "journey_id"
                        ],
                }
            )

    if not rows:

        return (
            pd.DataFrame()
        )

    data = (
        pd.DataFrame(
            rows
        )
    )

    return (
        data
        .groupby(
            [
                "from",
                "to",
            ],
            as_index=False,
        )
        .agg(
            journeys=(
                "journey_id",
                "nunique",
            )
        )
        .sort_values(
            "journeys",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

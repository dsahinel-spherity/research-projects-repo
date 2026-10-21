from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.services.umami_service import (
    get_real_analytics_data,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

MOCK_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "mock"
)


# =========================================================
# SYNTHETIC ENGAGEMENT DATA
# =========================================================

def load_mock_sessions():

    path = (
        MOCK_FOLDER
        / "mannstaedt_mock_sessions.csv"
    )

    df = pd.read_csv(
        path
    )

    if "date" in df.columns:

        df["date"] = (
            pd.to_datetime(
                df["date"],
                errors="coerce",
            )
        )

    return df


def load_mock_events():

    path = (
        MOCK_FOLDER
        / "mannstaedt_mock_events.csv"
    )

    df = pd.read_csv(
        path
    )

    if "date" in df.columns:

        df["date"] = (
            pd.to_datetime(
                df["date"],
                errors="coerce",
            )
        )

    return df


# =========================================================
# SYNTHETIC USER-JOURNEY / TRACEABILITY DATA
# ML ONLY
# =========================================================

def load_mock_journeys(
    rows=1200,
    random_state=42,
):

    rng = (
        np.random
        .default_rng(
            random_state
        )
    )

    sections = [
        "General Information",
        "Material Information",
        "Technical Data",
        "Sustainability",
        "Carbon Footprint",
        "Compliance",
        "Documents",
    ]

    devices = [
        "mobile",
        "laptop",
        "tablet",
    ]

    countries = [
        "DE",
        "NL",
        "FR",
        "BE",
        "AT",
        "PL",
    ]

    sources = (
        rng.choice(
            [
                "web",
                "qr",
            ],
            size=rows,
            p=[
                0.62,
                0.38,
            ],
        )
    )

    devices_array = (
        rng.choice(
            devices,
            size=rows,
            p=[
                0.48,
                0.46,
                0.06,
            ],
        )
    )

    country_array = (
        rng.choice(
            countries,
            size=rows,
            p=[
                0.62,
                0.08,
                0.08,
                0.05,
                0.07,
                0.10,
            ],
        )
    )

    first_sections = (
        rng.choice(
            sections,
            size=rows,
        )
    )

    section_count = (
        rng.poisson(
            2.7,
            size=rows,
        )
        + 1
    )

    section_count = (
        np.clip(
            section_count,
            1,
            7,
        )
    )

    repeated_sections = (
        rng.binomial(
            2,
            0.25,
            size=rows,
        )
    )

    unique_section_count = (
        np.maximum(
            1,
            section_count
            - repeated_sections,
        )
    )

    qr_entry = (
        sources
        == "qr"
    ).astype(int)

    mobile_entry = (
        devices_array
        == "mobile"
    ).astype(int)

    compliance_first = (
        first_sections
        == "Compliance"
    ).astype(int)

    documents_first = (
        first_sections
        == "Documents"
    ).astype(int)

    sustainability_first = (
        first_sections
        == "Sustainability"
    ).astype(int)

    duration_seconds = (
        18
        + section_count
        * 18
        + qr_entry
        * 10
        + mobile_entry
        * 4
        + sustainability_first
        * 9
        + rng.normal(
            0,
            25,
            size=rows,
        )
    )

    duration_seconds = (
        np.clip(
            duration_seconds,
            5,
            900,
        )
        .round(1)
    )

    document_probability = (
        0.04
        + section_count
        * 0.075
        + qr_entry
        * 0.08
        + documents_first
        * 0.24
        + compliance_first
        * 0.08
        + (
            duration_seconds
            >= 100
        ).astype(int)
        * 0.08
    )

    document_probability = (
        np.clip(
            document_probability,
            0.02,
            0.92,
        )
    )

    opened_document = (
        rng.binomial(
            1,
            document_probability,
        )
    )

    document_open_count = (
        opened_document
        * (
            1
            + rng.binomial(
                2,
                0.24,
                size=rows,
            )
        )
    )

    last_sections = (
        rng.choice(
            sections,
            size=rows,
        )
    )

    deep_engagement = (
        (
            section_count >= 4
        )
        | (
            document_open_count
            >= 1
        )
        | (
            duration_seconds
            >= 120
        )
    ).astype(int)

    journey_score = (
        section_count
        * 10
        + unique_section_count
        * 5
        + document_open_count
        * 25
        + np.minimum(
            duration_seconds,
            300,
        )
        / 10
    )

    journey_score = (
        np.clip(
            journey_score,
            0,
            100,
        )
        .round(1)
    )

    data = (
        pd.DataFrame(
            {
                "source":
                    sources,

                "qr_entry":
                    qr_entry,

                "device":
                    devices_array,

                "country":
                    country_array,

                "first_section":
                    first_sections,

                "last_section":
                    last_sections,

                "duration_seconds":
                    duration_seconds,

                "section_count":
                    section_count,

                "unique_section_count":
                    unique_section_count,

                "document_open_count":
                    document_open_count,

                "opened_document":
                    opened_document,

                "deep_engagement":
                    deep_engagement,

                "journey_score":
                    journey_score,
            }
        )
    )

    # -----------------------------------------------------
    # REALISTIC DATA QUALITY ISSUES
    # -----------------------------------------------------

    missing_devices = (
        rng.choice(
            data.index,
            size=max(
                1,
                int(
                    rows
                    * 0.015
                ),
            ),
            replace=False,
        )
    )

    data.loc[
        missing_devices,
        "device",
    ] = None

    missing_duration = (
        rng.choice(
            data.index,
            size=max(
                1,
                int(
                    rows
                    * 0.01
                ),
            ),
            replace=False,
        )
    )

    data.loc[
        missing_duration,
        "duration_seconds",
    ] = np.nan

    missing_country = (
        rng.choice(
            data.index,
            size=max(
                1,
                int(
                    rows
                    * 0.008
                ),
            ),
            replace=False,
        )
    )

    data.loc[
        missing_country,
        "country",
    ] = None

    duplicates = (
        data.sample(
            n=max(
                1,
                int(
                    rows
                    * 0.01
                ),
            ),
            random_state=
                random_state,
        )
    )

    data = (
        pd.concat(
            [
                data,
                duplicates,
            ],
            ignore_index=True,
        )
    )

    return data


# =========================================================
# REAL UMAMI DATA
# =========================================================

def load_real_analytics(
    start_at,
    end_at,
):

    raw = (
        get_real_analytics_data(
            start_at,
            end_at,
        )
    )

    events_df = (
        pd.DataFrame(
            raw.get(
                "events",
                [],
            )
        )
    )

    sessions_df = (
        pd.DataFrame(
            raw.get(
                "sessions",
                [],
            )
        )
    )

    pageview_points = (
        raw
        .get(
            "pageviews",
            {},
        )
        .get(
            "pageviews",
            [],
        )
    )

    pageviews_df = (
        pd.DataFrame(
            pageview_points
        )
    )

    if not events_df.empty:

        if (
            "createdAt"
            in events_df.columns
        ):

            events_df[
                "createdAt"
            ] = (
                pd.to_datetime(
                    events_df[
                        "createdAt"
                    ],
                    errors="coerce",
                    utc=True,
                )
            )

    if not sessions_df.empty:

        for column in [
            "firstAt",
            "lastAt",
        ]:

            if (
                column
                in sessions_df.columns
            ):

                sessions_df[
                    column
                ] = (
                    pd.to_datetime(
                        sessions_df[
                            column
                        ],
                        errors="coerce",
                        utc=True,
                    )
                )

    if not pageviews_df.empty:

        if (
            "x"
            in pageviews_df.columns
        ):

            pageviews_df[
                "date"
            ] = (
                pd.to_datetime(
                    pageviews_df[
                        "x"
                    ],
                    errors="coerce",
                    utc=True,
                )
            )

        if (
            "y"
            in pageviews_df.columns
        ):

            pageviews_df[
                "pageviews"
            ] = (
                pd.to_numeric(
                    pageviews_df[
                        "y"
                    ],
                    errors="coerce",
                )
                .fillna(0)
            )

    return {
        "stats":
            raw.get(
                "stats",
                {},
            ),

        "events":
            events_df,

        "sessions":
            sessions_df,

        "pageviews":
            pageviews_df,

        "devices":
            pd.DataFrame(
                raw.get(
                    "devices",
                    [],
                )
            ),

        "countries":
            pd.DataFrame(
                raw.get(
                    "countries",
                    [],
                )
            ),

        "regions":
            pd.DataFrame(
                raw.get(
                    "regions",
                    [],
                )
            ),

        "paths":
            pd.DataFrame(
                raw.get(
                    "paths",
                    [],
                )
            ),
    }


# =========================================================
# ENGAGEMENT ML PREPARATION
# =========================================================

def prepare_engagement_ml_data(
    df,
):

    result = (
        df.copy()
    )

    categorical_columns = [
        column
        for column
        in [
            "device",
            "region",
            "source",
        ]
        if column
        in result.columns
    ]

    if categorical_columns:

        result = (
            pd.get_dummies(
                result,
                columns=
                    categorical_columns,
                dtype=int,
            )
        )

    drop_columns = [
        "date",
        "session_id",
        "dpp_id",
        "data_origin",
    ]

    result = (
        result.drop(
            columns=[
                column
                for column
                in drop_columns
                if column
                in result.columns
            ]
        )
    )

    return result


# =========================================================
# JOURNEY ML PREPARATION
# =========================================================

def prepare_journey_ml_data(
    df,
):

    result = (
        df.copy()
    )

    categorical_columns = [
        column
        for column
        in [
            "source",
            "device",
            "country",
            "first_section",
            "last_section",
        ]
        if column
        in result.columns
    ]

    if categorical_columns:

        result = (
            pd.get_dummies(
                result,
                columns=
                    categorical_columns,
                dtype=int,
            )
        )

    return result


# =========================================================
# DATASET REGISTRY
# =========================================================

def get_available_datasets():

    return {
        "Synthetic Engagement Sessions":
            "mock_sessions",

        "Synthetic DPP Events":
            "mock_events",

        "Synthetic User Journeys":
            "mock_journeys",
    }


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(
    dataset_key,
):

    if (
        dataset_key
        == "mock_sessions"
    ):

        return (
            load_mock_sessions()
        )

    if (
        dataset_key
        == "mock_events"
    ):

        return (
            load_mock_events()
        )

    if (
        dataset_key
        == "mock_journeys"
    ):

        return (
            load_mock_journeys()
        )

    raise ValueError(
        f"Unknown synthetic dataset: "
        f"{dataset_key}"
    )
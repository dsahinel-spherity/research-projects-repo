from pathlib import Path

import pandas as pd


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
# RAW MOCK DATA
# =========================================================

def load_mock_sessions():

    path = (
        MOCK_FOLDER
        / "mannstaedt_mock_sessions.csv"
    )

    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"]
        )

    return df


def load_mock_events():

    path = (
        MOCK_FOLDER
        / "mannstaedt_mock_events.csv"
    )

    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"]
        )

    return df


def load_mock_traceability():

    path = (
        MOCK_FOLDER
        / "mannstaedt_mock_traceability.csv"
    )

    return pd.read_csv(path)


# =========================================================
# ENGAGEMENT ML DATA
# =========================================================

def prepare_engagement_ml_data(
    df,
):

    result = df.copy()

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

        result = pd.get_dummies(
            result,
            columns=categorical_columns,
            dtype=int,
        )

    drop_columns = [
        "date",
        "session_id",
        "dpp_id",
        "data_origin",
    ]

    result = result.drop(
        columns=[
            column
            for column
            in drop_columns
            if column
            in result.columns
        ]
    )

    return result


# =========================================================
# TRACEABILITY ML DATA
# =========================================================

def prepare_traceability_ml_data(
    df,
):

    result = df.copy()

    # -----------------------------------------------------
    # HIGH-RISK CLASSIFICATION TARGET
    # -----------------------------------------------------

    if "risk_score" in result.columns:

        result[
            "high_risk"
        ] = (
            result[
                "risk_score"
            ]
            >= 0.70
        ).astype(int)

    # -----------------------------------------------------
    # CATEGORICAL FEATURES
    # -----------------------------------------------------

    categorical_columns = [
        column
        for column
        in [
            "supplier_id",
            "material",
            "origin_country",
            "credential_status",
        ]
        if column
        in result.columns
    ]

    if categorical_columns:

        result = pd.get_dummies(
            result,
            columns=categorical_columns,
            dtype=int,
        )

    # -----------------------------------------------------
    # REMOVE IDENTIFIERS
    # -----------------------------------------------------

    drop_columns = [
        "product_id",
        "batch_id",
        "data_origin",
    ]

    result = result.drop(
        columns=[
            column
            for column
            in drop_columns
            if column
            in result.columns
        ]
    )

    return result


# =========================================================
# SUPPLIER RISK PROFILES
# =========================================================

def prepare_supplier_risk_data():

    traceability = (
        load_mock_traceability()
    )

    traceability[
        "credential_issue"
    ] = (
        traceability[
            "credential_status"
        ]
        != "valid"
    ).astype(int)

    supplier_df = (
        traceability
        .groupby(
            "supplier_id"
        )
        .agg(

            records=(
                "supplier_id",
                "size",
            ),

            average_risk=(
                "risk_score",
                "mean",
            ),

            missing_link_rate=(
                "missing_link",
                "mean",
            ),

            credential_issue_rate=(
                "credential_issue",
                "mean",
            ),
        )
        .reset_index()
    )

    return supplier_df


# =========================================================
# SECTION PERFORMANCE
# =========================================================

def prepare_section_performance_data():

    events = (
        load_mock_events()
    )

    section_events = (
        events[
            events[
                "event_name"
            ]
            == "dpp-section-view"
        ]
        .copy()
    )

    if section_events.empty:

        return pd.DataFrame(
            columns=[
                "section_name",
                "views",
                "unique_sessions",
                "view_share",
                "underused",
            ]
        )

    if (
        "session_id"
        in section_events.columns
    ):

        section_df = (
            section_events
            .groupby(
                "section_name"
            )
            .agg(

                views=(
                    "section_name",
                    "size",
                ),

                unique_sessions=(
                    "session_id",
                    "nunique",
                ),
            )
            .reset_index()
        )

    else:

        section_df = (
            section_events
            .groupby(
                "section_name"
            )
            .size()
            .rename(
                "views"
            )
            .reset_index()
        )

        section_df[
            "unique_sessions"
        ] = (
            section_df[
                "views"
            ]
        )

    total_views = (
        section_df[
            "views"
        ].sum()
    )

    if total_views > 0:

        section_df[
            "view_share"
        ] = (
            section_df[
                "views"
            ]
            / total_views
        )

    else:

        section_df[
            "view_share"
        ] = 0.0

    median_views = (
        section_df[
            "views"
        ].median()
    )

    section_df[
        "underused"
    ] = (
        section_df[
            "views"
        ]
        < median_views
    )

    return (
        section_df
        .sort_values(
            "views",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# DATASET REGISTRY
# =========================================================

def get_available_datasets():

    return {

        "Synthetic Engagement Sessions":
            "mock_sessions",

        "Synthetic DPP Events":
            "mock_events",

        "Synthetic Traceability":
            "mock_traceability",
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

        return load_mock_sessions()

    if (
        dataset_key
        == "mock_events"
    ):

        return load_mock_events()

    if (
        dataset_key
        == "mock_traceability"
    ):

        return load_mock_traceability()

    raise ValueError(
        f"Unknown dataset: {dataset_key}"
    )
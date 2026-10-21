from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.data_service import (
    prepare_engagement_ml_data,
    prepare_journey_ml_data,
)


# =========================================================
# HELPERS
# =========================================================

def _safe_median(
    series,
):

    numeric = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
    )

    median = (
        numeric.median()
    )

    if pd.isna(
        median
    ):

        return 0.0

    return float(
        median
    )


# =========================================================
# BASE CLEANING
# =========================================================

def _clean_base_dataframe(
    raw_df,
):

    data = (
        raw_df.copy()
    )

    original_rows = (
        len(
            data
        )
    )

    original_columns = (
        data.columns
        .tolist()
    )

    duplicate_count = int(
        data
        .duplicated()
        .sum()
    )

    data = (
        data
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    numeric_columns = (
        data
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    infinite_values_cleaned = 0

    for column in (
        numeric_columns
    ):

        mask = (
            data[
                column
            ]
            .isin(
                [
                    np.inf,
                    -np.inf,
                ]
            )
        )

        infinite_values_cleaned += int(
            mask.sum()
        )

        data.loc[
            mask,
            column,
        ] = np.nan

    columns_with_missing = [
        column
        for column
        in data.columns
        if (
            data[
                column
            ]
            .isna()
            .any()
        )
    ]

    missing_values_filled = 0

    for column in (
        data.columns
    ):

        missing_count = int(
            data[
                column
            ]
            .isna()
            .sum()
        )

        if (
            missing_count
            == 0
        ):

            continue

        if (
            pd.api.types
            .is_numeric_dtype(
                data[
                    column
                ]
            )
        ):

            fill_value = (
                _safe_median(
                    data[
                        column
                    ]
                )
            )

        elif (
            pd.api.types
            .is_datetime64_any_dtype(
                data[
                    column
                ]
            )
        ):

            continue

        else:

            mode = (
                data[
                    column
                ]
                .mode(
                    dropna=True
                )
            )

            fill_value = (
                mode.iloc[
                    0
                ]
                if not mode.empty
                else "unknown"
            )

        data[
            column
        ] = (
            data[
                column
            ]
            .fillna(
                fill_value
            )
        )

        missing_values_filled += (
            missing_count
        )

    summary = {
        "original_rows":
            original_rows,

        "original_columns":
            original_columns,

        "duplicates_removed":
            duplicate_count,

        "infinite_values_cleaned":
            int(
                infinite_values_cleaned
            ),

        "columns_with_missing":
            columns_with_missing,

        "missing_values_filled":
            int(
                missing_values_filled
            ),
    }

    return (
        data,
        summary,
    )


# =========================================================
# PREPARE DATASET
# =========================================================

def prepare_dataset_for_ml(
    dataset_key,
    raw_df,
):

    working_df, summary = (
        _clean_base_dataframe(
            raw_df
        )
    )

    # Human-readable cleaned version.
    cleaned_dataframe = (
        working_df.copy()
        .reset_index(
            drop=True
        )
    )

    encoded_columns = []

    excluded_columns = []

    # =====================================================
    # ENGAGEMENT SESSIONS
    # =====================================================

    if (
        dataset_key
        == "mock_sessions"
    ):

        encoded_columns = [
            column
            for column
            in [
                "device",
                "region",
                "source",
            ]
            if column
            in working_df.columns
        ]

        excluded_columns = [
            column
            for column
            in [
                "date",
                "session_id",
                "dpp_id",
                "data_origin",
            ]
            if column
            in working_df.columns
        ]

        prepared = (
            prepare_engagement_ml_data(
                working_df
            )
        )

    # =====================================================
    # SYNTHETIC USER JOURNEYS
    # =====================================================

    elif (
        dataset_key
        == "mock_journeys"
    ):

        encoded_columns = [
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
            in working_df.columns
        ]

        prepared = (
            prepare_journey_ml_data(
                working_df
            )
        )

    # =====================================================
    # SYNTHETIC EVENTS
    # =====================================================

    elif (
        dataset_key
        == "mock_events"
    ):

        prepared = (
            working_df.copy()
        )

        identifier_columns = [
            column
            for column
            in [
                "session_id",
                "dpp_id",
                "document_name",
                "section_name",
                "data_origin",
            ]
            if column
            in prepared.columns
        ]

        excluded_columns.extend(
            identifier_columns
        )

        if identifier_columns:

            prepared = (
                prepared.drop(
                    columns=
                        identifier_columns
                )
            )

        if (
            "date"
            in prepared.columns
        ):

            excluded_columns.append(
                "date"
            )

            prepared = (
                prepared.drop(
                    columns=[
                        "date"
                    ]
                )
            )

        categorical_columns = [
            column
            for column
            in prepared.columns
            if (
                prepared[
                    column
                ].dtype
                == "object"
            )
        ]

        encoded_columns.extend(
            categorical_columns
        )

        if categorical_columns:

            prepared = (
                pd.get_dummies(
                    prepared,
                    columns=
                        categorical_columns,
                    dtype=int,
                )
            )

    # =====================================================
    # GENERIC FALLBACK
    # =====================================================

    else:

        prepared = (
            working_df.copy()
        )

        categorical_columns = [
            column
            for column
            in prepared.columns
            if (
                prepared[
                    column
                ].dtype
                == "object"
            )
        ]

        encoded_columns.extend(
            categorical_columns
        )

        if categorical_columns:

            prepared = (
                pd.get_dummies(
                    prepared,
                    columns=
                        categorical_columns,
                    dtype=int,
                )
            )

    # =====================================================
    # REMOVE DATETIME COLUMNS
    # =====================================================

    datetime_columns = [
        column
        for column
        in prepared.columns
        if (
            pd.api.types
            .is_datetime64_any_dtype(
                prepared[
                    column
                ]
            )
        )
    ]

    if datetime_columns:

        excluded_columns.extend(
            datetime_columns
        )

        prepared = (
            prepared.drop(
                columns=
                    datetime_columns
            )
        )

    # =====================================================
    # REMOVE NON NUMERIC REMAINDERS
    # =====================================================

    non_numeric_columns = [
        column
        for column
        in prepared.columns
        if not (
            pd.api.types
            .is_numeric_dtype(
                prepared[
                    column
                ]
            )
        )
    ]

    if non_numeric_columns:

        excluded_columns.extend(
            non_numeric_columns
        )

        prepared = (
            prepared.drop(
                columns=
                    non_numeric_columns
            )
        )

    # =====================================================
    # FINAL NUMERIC CLEANING
    # =====================================================

    prepared = (
        prepared.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    remaining_missing = int(
        prepared
        .isna()
        .sum()
        .sum()
    )

    if (
        remaining_missing
        > 0
    ):

        for column in (
            prepared.columns
        ):

            if (
                prepared[
                    column
                ]
                .isna()
                .any()
            ):

                prepared[
                    column
                ] = (
                    prepared[
                        column
                    ]
                    .fillna(
                        _safe_median(
                            prepared[
                                column
                            ]
                        )
                    )
                )

        summary[
            "missing_values_filled"
        ] += remaining_missing

    prepared = (
        prepared
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary[
        "final_rows"
    ] = (
        len(
            prepared
        )
    )

    summary[
        "rows_removed"
    ] = (
        summary[
            "original_rows"
        ]
        - summary[
            "final_rows"
        ]
    )

    summary[
        "encoded_columns"
    ] = sorted(
        set(
            encoded_columns
        )
    )

    summary[
        "excluded_columns"
    ] = sorted(
        set(
            excluded_columns
        )
    )

    summary[
        "final_columns"
    ] = (
        len(
            prepared.columns
        )
    )

    summary[
        "model_columns"
    ] = (
        prepared.columns
        .tolist()
    )

    return {
        "dataframe":
            prepared,

        "cleaned_dataframe":
            cleaned_dataframe,

        "summary":
            summary,
    }
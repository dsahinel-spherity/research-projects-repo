import numpy as np
import pandas as pd

from app.services.data_service import (
    prepare_engagement_ml_data,
    prepare_traceability_ml_data,
)


# =========================================================
# DATASET RULES
# =========================================================

DATASET_RULES = {

    "mock_sessions": {

        "encoded_columns": [
            "device",
            "region",
            "source",
        ],

        "excluded_columns": [
            "date",
            "session_id",
            "dpp_id",
            "data_origin",
        ],
    },


    "mock_traceability": {

        "encoded_columns": [
            "supplier_id",
            "material",
            "origin_country",
            "credential_status",
        ],

        "excluded_columns": [
            "product_id",
            "batch_id",
            "data_origin",
        ],
    },


    "mock_events": {

        "encoded_columns": [],

        "excluded_columns": [],
    },
}


# =========================================================
# PREPARE DATASET
# =========================================================

def prepare_dataset_for_ml(
    dataset_key: str,
    raw_df: pd.DataFrame,
):
    """
    Prepare a selected dataset for AnalyticEngine use.

    This function is intentionally transparent:
    it returns both the prepared DataFrame and
    a human-readable summary of what changed.
    """

    if raw_df is None:

        raise ValueError(
            "Dataset cannot be None."
        )


    working_df = (
        raw_df.copy()
    )


    # =====================================================
    # INITIAL INFORMATION
    # =====================================================

    original_rows = len(
        working_df
    )

    original_columns = len(
        working_df.columns
    )

    original_column_names = (
        working_df.columns
        .tolist()
    )


    original_missing_values = int(
        working_df
        .isna()
        .sum()
        .sum()
    )


    # =====================================================
    # DUPLICATES
    # =====================================================

    duplicates_found = int(
        working_df
        .duplicated()
        .sum()
    )


    if duplicates_found > 0:

        working_df = (
            working_df
            .drop_duplicates()
            .reset_index(
                drop=True
            )
        )


    # =====================================================
    # INFINITE VALUES
    # =====================================================

    numeric_before = (
        working_df
        .select_dtypes(
            include=[
                "number"
            ]
        )
    )


    infinite_values_found = 0


    if not numeric_before.empty:

        infinite_values_found = int(
            np.isinf(
                numeric_before
                .to_numpy(
                    dtype=float
                )
            )
            .sum()
        )


    if infinite_values_found > 0:

        numeric_columns = (
            working_df
            .select_dtypes(
                include=[
                    "number"
                ]
            )
            .columns
        )


        working_df[
            numeric_columns
        ] = (
            working_df[
                numeric_columns
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )


    # =====================================================
    # DATASET-SPECIFIC PREPARATION
    # =====================================================

    rules = (
        DATASET_RULES.get(
            dataset_key,
            {
                "encoded_columns": [],
                "excluded_columns": [],
            },
        )
    )


    encoded_columns = [
        column

        for column
        in rules[
            "encoded_columns"
        ]

        if column
        in working_df.columns
    ]


    excluded_columns = [
        column

        for column
        in rules[
            "excluded_columns"
        ]

        if column
        in working_df.columns
    ]


    if (
        dataset_key
        == "mock_sessions"
    ):

        prepared_df = (
            prepare_engagement_ml_data(
                working_df
            )
        )


    elif (
        dataset_key
        == "mock_traceability"
    ):

        prepared_df = (
            prepare_traceability_ml_data(
                working_df
            )
        )


    else:

        # Generic fallback.
        #
        # AnalyticEngine algorithms currently work
        # with numeric matrices, so generic datasets
        # expose only numeric columns.

        prepared_df = (
            working_df
            .select_dtypes(
                include=[
                    "number",
                    "bool",
                ]
            )
            .copy()
        )


        excluded_columns = [
            column

            for column
            in working_df.columns

            if column
            not in prepared_df.columns
        ]


    # =====================================================
    # BOOL -> INT
    # =====================================================

    bool_columns = (
        prepared_df
        .select_dtypes(
            include=[
                "bool"
            ]
        )
        .columns
        .tolist()
    )


    for column in bool_columns:

        prepared_df[
            column
        ] = (
            prepared_df[
                column
            ]
            .astype(int)
        )


    # =====================================================
    # ENSURE NUMERIC MODEL MATRIX
    # =====================================================

    non_numeric_remaining = [
        column

        for column
        in prepared_df.columns

        if not pd.api.types.is_numeric_dtype(
            prepared_df[
                column
            ]
        )
    ]


    if non_numeric_remaining:

        prepared_df = (
            prepared_df.drop(
                columns=
                    non_numeric_remaining
            )
        )


        for column in non_numeric_remaining:

            if (
                column
                not in excluded_columns
            ):

                excluded_columns.append(
                    column
                )


    # =====================================================
    # MISSING NUMERIC VALUES
    # =====================================================

    missing_before_fill = int(
        prepared_df
        .isna()
        .sum()
        .sum()
    )


    columns_with_missing = []


    for column in (
        prepared_df.columns
    ):

        missing_count = int(
            prepared_df[
                column
            ]
            .isna()
            .sum()
        )


        if missing_count == 0:

            continue


        columns_with_missing.append(
            column
        )


        median_value = (
            prepared_df[
                column
            ]
            .median()
        )


        if pd.isna(
            median_value
        ):

            median_value = 0


        prepared_df[
            column
        ] = (
            prepared_df[
                column
            ]
            .fillna(
                median_value
            )
        )


    # =====================================================
    # FINAL INFINITE CHECK
    # =====================================================

    prepared_df = (
        prepared_df
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )


    remaining_missing = int(
        prepared_df
        .isna()
        .sum()
        .sum()
    )


    if remaining_missing:

        prepared_df = (
            prepared_df
            .fillna(0)
        )


    # =====================================================
    # FINAL INFORMATION
    # =====================================================

    final_rows = len(
        prepared_df
    )

    final_columns = len(
        prepared_df.columns
    )


    rows_removed = (
        original_rows
        - final_rows
    )


    generated_columns = [
        column

        for column
        in prepared_df.columns

        if column
        not in original_column_names
    ]


    summary = {

        "dataset_key":
            dataset_key,

        "original_rows":
            original_rows,

        "final_rows":
            final_rows,

        "rows_removed":
            rows_removed,

        "original_columns":
            original_columns,

        "final_columns":
            final_columns,

        "duplicates_removed":
            duplicates_found,

        "original_missing_values":
            original_missing_values,

        "missing_values_filled":
            missing_before_fill,

        "columns_with_missing":
            columns_with_missing,

        "infinite_values_cleaned":
            infinite_values_found,

        "encoded_columns":
            encoded_columns,

        "excluded_columns":
            excluded_columns,

        "generated_columns":
            generated_columns,

        "model_columns":
            prepared_df.columns.tolist(),
    }


    return {

        "dataframe":
            prepared_df,

        "summary":
            summary,
    }
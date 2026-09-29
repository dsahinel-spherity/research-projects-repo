from datetime import timedelta

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. CLEAN RAW TRAFFIC DATA
# ---------------------------------------------------------

def clean_company_traffic(df: pd.DataFrame):
    """
    Clean the aggregated company traffic before ML.

    Steps:
    - make sure dates are valid
    - make sure views are numeric
    - remove invalid rows
    - aggregate duplicate dates
    - create missing calendar days
    - fill missing days with 0 views
    - remove impossible negative values
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "pageviews",
            ]
        )

    data = df.copy()

    # Rename from analytics layer format
    if "views" in data.columns:
        data = data.rename(
            columns={
                "views": "pageviews"
            }
        )

    # Convert date
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
        utc=True,
    )

    # Convert traffic to numeric
    data["pageviews"] = pd.to_numeric(
        data["pageviews"],
        errors="coerce",
    )

    # Drop unusable records
    data = data.dropna(
        subset=[
            "date",
            "pageviews",
        ]
    )

    # Traffic cannot be negative
    data["pageviews"] = (
        data["pageviews"]
        .clip(lower=0)
    )

    # If several records exist for same date,
    # combine them.
    data = (
        data.groupby(
            "date",
            as_index=False,
        )["pageviews"]
        .sum()
        .sort_values("date")
    )

    if data.empty:
        return data

    # -----------------------------------------------------
    # Add missing calendar days
    # -----------------------------------------------------

    full_dates = pd.date_range(
        start=data["date"].min(),
        end=data["date"].max(),
        freq="D",
        tz="UTC",
    )

    data = (
        data.set_index("date")
        .reindex(full_dates)
        .rename_axis("date")
        .reset_index()
    )

    # A missing Umami daily record means 0 recorded views.
    data["pageviews"] = (
        data["pageviews"]
        .fillna(0)
    )

    data = data.reset_index(
        drop=True
    )

    return data


# ---------------------------------------------------------
# 2. FEATURE CREATION
# ---------------------------------------------------------

def create_ml_features(
    cleaned_df: pd.DataFrame,
):
    """
    Create features from historical traffic.

    Features:
    - previous day
    - two days ago
    - seven days ago
    - previous 3-day average
    - previous 7-day average
    - day of week
    """

    data = cleaned_df.copy()

    data["lag_1"] = (
        data["pageviews"]
        .shift(1)
    )

    data["lag_2"] = (
        data["pageviews"]
        .shift(2)
    )

    data["lag_7"] = (
        data["pageviews"]
        .shift(7)
    )

    data["rolling_3"] = (
        data["pageviews"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    data["rolling_7"] = (
        data["pageviews"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    data["day_of_week"] = (
        data["date"]
        .dt.dayofweek
    )

    # Rows at the beginning cannot have
    # lag/rolling features yet.
    data = (
        data.dropna()
        .reset_index(drop=True)
    )

    return data


# ---------------------------------------------------------
# 3. NORMALIZATION
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_7",
    "rolling_3",
    "rolling_7",
    "day_of_week",
]


def normalize_features(
    feature_df: pd.DataFrame,
    scaler=None,
):
    """
    Normalize the ML input features.

    Returns:
    - scaled dataframe
    - fitted StandardScaler
    """

    if feature_df.empty:
        return pd.DataFrame(), scaler

    X = feature_df[
        FEATURE_COLUMNS
    ].copy()

    if scaler is None:
        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            X
        )

    else:
        scaled = scaler.transform(
            X
        )

    scaled_df = pd.DataFrame(
        scaled,
        columns=FEATURE_COLUMNS,
        index=X.index,
    )

    return scaled_df, scaler


# ---------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------

def train_traffic_model(
    cleaned_df: pd.DataFrame,
):
    """
    Train Random Forest with chronological validation.

    The last 20% of history is used as validation.
    """

    feature_df = create_ml_features(
        cleaned_df
    )

    # Require enough historical observations.
    if len(feature_df) < 14:
        return {
            "model": None,
            "scaler": None,
            "mae": None,
            "feature_df": feature_df,
            "normalized_df": pd.DataFrame(),
            "validation_actual": [],
            "validation_predicted": [],
        }

    X = feature_df[
        FEATURE_COLUMNS
    ]

    y = feature_df[
        "pageviews"
    ]

    test_size = max(
        3,
        int(
            len(feature_df)
            * 0.20
        ),
    )

    X_train = X.iloc[
        :-test_size
    ]

    y_train = y.iloc[
        :-test_size
    ]

    X_test = X.iloc[
        -test_size:
    ]

    y_test = y.iloc[
        -test_size:
    ]

    # -----------------------------------------------------
    # Normalize using TRAINING data only
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    validation_predictions = (
        model.predict(
            X_test_scaled
        )
    )

    mae = mean_absolute_error(
        y_test,
        validation_predictions,
    )

    # -----------------------------------------------------
    # Retrain final model using ALL data
    # -----------------------------------------------------

    final_scaler = StandardScaler()

    X_all_scaled = (
        final_scaler.fit_transform(
            X
        )
    )

    model.fit(
        X_all_scaled,
        y,
    )

    normalized_df = pd.DataFrame(
        X_all_scaled,
        columns=FEATURE_COLUMNS,
    )

    normalized_df.insert(
        0,
        "date",
        feature_df[
            "date"
        ].reset_index(
            drop=True
        ),
    )

    normalized_df[
        "target_pageviews"
    ] = (
        y.reset_index(
            drop=True
        )
    )

    return {
        "model": model,
        "scaler": final_scaler,
        "mae": round(
            mae,
            2,
        ),
        "feature_df": feature_df,
        "normalized_df": normalized_df,
        "validation_actual": list(
            y_test
        ),
        "validation_predicted": list(
            validation_predictions
        ),
    }


# ---------------------------------------------------------
# 5. FORECAST
# ---------------------------------------------------------

def forecast_next_days(
    cleaned_df,
    model,
    scaler,
    days=7,
):
    """
    Predict traffic recursively for the next N days.
    """

    if (
        model is None
        or scaler is None
        or cleaned_df.empty
    ):
        return pd.DataFrame()

    history = (
        cleaned_df.copy()
    )

    predictions = []

    for _ in range(days):

        next_date = (
            history["date"].max()
            + timedelta(days=1)
        )

        lag_1 = history.iloc[
            -1
        ]["pageviews"]

        lag_2 = history.iloc[
            -2
        ]["pageviews"]

        lag_7 = history.iloc[
            -7
        ]["pageviews"]

        rolling_3 = (
            history["pageviews"]
            .tail(3)
            .mean()
        )

        rolling_7 = (
            history["pageviews"]
            .tail(7)
            .mean()
        )

        day_of_week = (
            next_date.dayofweek
        )

        next_features = (
            pd.DataFrame(
                [
                    {
                        "lag_1": lag_1,
                        "lag_2": lag_2,
                        "lag_7": lag_7,
                        "rolling_3": rolling_3,
                        "rolling_7": rolling_7,
                        "day_of_week": day_of_week,
                    }
                ]
            )
        )

        next_scaled = (
            scaler.transform(
                next_features[
                    FEATURE_COLUMNS
                ]
            )
        )

        prediction = (
            model.predict(
                next_scaled
            )[0]
        )

        prediction = max(
            0,
            round(prediction),
        )

        predictions.append(
            {
                "date": next_date,
                "predicted_pageviews":
                    prediction,
            }
        )

        new_row = pd.DataFrame(
            [
                {
                    "date": next_date,
                    "pageviews":
                        prediction,
                }
            ]
        )

        history = pd.concat(
            [
                history,
                new_row,
            ],
            ignore_index=True,
        )

    return pd.DataFrame(
        predictions
    )


# ---------------------------------------------------------
# COMPLETE ML PIPELINE
# ---------------------------------------------------------

def build_company_forecast(
    company_traffic_df,
    days=7,
):
    """
    Complete company-specific ML workflow.
    """

    cleaned = (
        clean_company_traffic(
            company_traffic_df
        )
    )

    training = (
        train_traffic_model(
            cleaned
        )
    )

    if (
        training["model"]
        is None
    ):

        return {
            "status":
                "not_enough_data",

            "cleaned":
                cleaned,

            "features":
                training[
                    "feature_df"
                ],

            "normalized":
                training[
                    "normalized_df"
                ],

            "forecast":
                pd.DataFrame(),

            "mae":
                None,
        }

    forecast = (
        forecast_next_days(
            cleaned,
            training["model"],
            training["scaler"],
            days=days,
        )
    )

    return {
        "status": "ok",

        "cleaned":
            cleaned,

        "features":
            training[
                "feature_df"
            ],

        "normalized":
            training[
                "normalized_df"
            ],

        "forecast":
            forecast,

        "mae":
            training[
                "mae"
            ],
    }
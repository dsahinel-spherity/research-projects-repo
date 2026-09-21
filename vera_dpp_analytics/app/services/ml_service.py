from datetime import timedelta

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def prepare_pageview_dataframe(pageviews_data: dict):
    """
    Convert Umami pageviews response into a clean dataframe.

    Expected Umami format:
    {
        "pageviews": [
            {"x": "2026-08-10T00:00:00Z", "y": 10},
            ...
        ]
    }
    """

    pageviews = pageviews_data.get("pageviews", [])

    if not pageviews:
        return pd.DataFrame(columns=["date", "pageviews"])

    df = pd.DataFrame(pageviews)

    df = df.rename(
        columns={
            "x": "date",
            "y": "pageviews",
        }
    )

    df["date"] = pd.to_datetime(df["date"])
    df["pageviews"] = pd.to_numeric(
        df["pageviews"],
        errors="coerce",
    ).fillna(0)

    df = df.sort_values("date").reset_index(drop=True)

    return df


def create_ml_features(df: pd.DataFrame):
    """
    Create simple time-series features.

    We use:
    - yesterday's pageviews
    - pageviews two days ago
    - 3-day rolling average
    - day of week
    """

    data = df.copy()

    data["lag_1"] = data["pageviews"].shift(1)
    data["lag_2"] = data["pageviews"].shift(2)

    data["rolling_3"] = (
        data["pageviews"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    data["day_of_week"] = data["date"].dt.dayofweek

    data = data.dropna().reset_index(drop=True)

    return data


def train_traffic_model(df: pd.DataFrame):
    """
    Train a Random Forest model.

    Returns the trained model plus a simple MAE evaluation
    when enough historical data is available.
    """

    feature_df = create_ml_features(df)

    if len(feature_df) < 7:
        return None, None

    features = [
        "lag_1",
        "lag_2",
        "rolling_3",
        "day_of_week",
    ]

    X = feature_df[features]
    y = feature_df["pageviews"]

    # Small time-based test split.
    # We do not randomly shuffle time-series observations.
    test_size = max(2, int(len(feature_df) * 0.2))

    if len(feature_df) > test_size + 3:
        X_train = X.iloc[:-test_size]
        y_train = y.iloc[:-test_size]

        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]
    else:
        X_train = X
        y_train = y

        X_test = None
        y_test = None

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )

    model.fit(X_train, y_train)

    mae = None

    if X_test is not None:
        predictions = model.predict(X_test)

        mae = round(
            mean_absolute_error(
                y_test,
                predictions,
            ),
            2,
        )

        # Retrain on all historical data after evaluation.
        model.fit(X, y)

    return model, mae


def forecast_next_days(
    df: pd.DataFrame,
    model,
    days: int = 7,
):
    """
    Iteratively predict future pageviews.

    Each prediction becomes part of the history used
    to predict the following day.
    """

    if model is None or df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "predicted_pageviews",
            ]
        )

    history = df.copy()

    predictions = []

    for _ in range(days):

        next_date = (
            history["date"].max()
            + timedelta(days=1)
        )

        lag_1 = history.iloc[-1]["pageviews"]

        if len(history) >= 2:
            lag_2 = history.iloc[-2]["pageviews"]
        else:
            lag_2 = lag_1

        rolling_3 = (
            history["pageviews"]
            .tail(3)
            .mean()
        )

        day_of_week = next_date.dayofweek

        feature_row = pd.DataFrame(
            [
                {
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "rolling_3": rolling_3,
                    "day_of_week": day_of_week,
                }
            ]
        )

        predicted_value = model.predict(
            feature_row
        )[0]

        # Traffic cannot be negative.
        predicted_value = max(
            0,
            round(predicted_value),
        )

        predictions.append(
            {
                "date": next_date,
                "predicted_pageviews": predicted_value,
            }
        )

        new_row = pd.DataFrame(
            [
                {
                    "date": next_date,
                    "pageviews": predicted_value,
                }
            ]
        )

        history = pd.concat(
            [history, new_row],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


def build_traffic_forecast(
    pageviews_data: dict,
    days: int = 7,
):
    """
    Complete ML workflow used by the dashboard.
    """

    df = prepare_pageview_dataframe(
        pageviews_data
    )

    model, mae = train_traffic_model(df)

    if model is None:
        return {
            "historical": df,
            "forecast": pd.DataFrame(),
            "mae": None,
            "status": "not_enough_data",
        }

    forecast = forecast_next_days(
        df,
        model,
        days=days,
    )

    return {
        "historical": df,
        "forecast": forecast,
        "mae": mae,
        "status": "ok",
    }
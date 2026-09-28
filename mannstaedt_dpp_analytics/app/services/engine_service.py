import tempfile

from pathlib import Path

import pandas as pd

from analytics_engine.engine import (
    AnalyticEngine,
)


MODEL_OPTIONS = {

    "Linear Regression":
        "linear-regression",

    "Logistic Regression":
        "logistic-regression",

    "Random Forest Regression":
        "random-forest-regression",

    "Random Forest Classification":
        "random-forest-classification",

    "KMeans Clustering":
        "cluster",

    "Isolation Forest":
        "isolation-forest",

    "Feature Selection":
        "feature-selection",
}


FUNCTION_OPTIONS = {

    "linear-regression": [
        "predict",
        "result",
        "predict_input",
    ],

    "logistic-regression": [
        "predict",
        "result",
        "confusion",
        "predict_input",
    ],

    "random-forest-regression": [
        "predict",
        "result",
        "predict_input",
    ],

    "random-forest-classification": [
        "predict",
        "confusion",
        "predict_input",
    ],

    "cluster": [
        "predict",
        "result",
        "elbow",
        "silhouette",
    ],

    "isolation-forest": [
        "predict",
        "result",
    ],

    "feature-selection": [
        "correlation",
        "correlate",
        "correlated",
        "pca_info",
        "anova_regression",
        "mutual_info_regression",
        "lasso",
    ],
}


def run_engine(
    dataframe: pd.DataFrame,
    algorithm_name: str,
    function_name: str,
    included_columns=None,
    target_columns=None,
    config=None,
    function_input=None,
):

    included_columns = (
        included_columns
        or []
    )


    target_columns = (
        target_columns
        or []
    )


    config = (
        config
        or {}
    )


    function_input = (
        function_input
        or {}
    )


    with tempfile.TemporaryDirectory() as tmp:

        file_name = (
            "selected_dataset.csv"
        )


        file_path = (
            Path(tmp)
            / file_name
        )


        dataframe.to_csv(
            file_path,
            index=False,
        )


        setup_data = {

            "algorithm": {

                "name":
                    algorithm_name,

                "function":
                    function_name,

                "input":
                    function_input,

                "config":
                    config,
            },

            "data": {

                "file": {

                    "file_name":
                        file_name,

                    "path":
                        tmp,
                }
            },

            "features":
                {},
        }


        if included_columns:

            setup_data[
                "features"
            ][
                "included"
            ] = (
                included_columns
            )


        if target_columns:

            setup_data[
                "features"
            ][
                "target"
            ] = (
                target_columns
            )


        result = (
            AnalyticEngine(
                setup_data
            )
            .handle()
        )


        return {

            "setup_data":
                setup_data,

            "result":
                result,
        }
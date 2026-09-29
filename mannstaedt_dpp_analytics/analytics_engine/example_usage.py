"""
example_usage.py
=================

Minimal, runnable demonstration of AnalyticEngine. Run this file directly:

    python -m analytics_engine.example_usage

It creates a small synthetic CSV (standing in for, e.g., daily website
traffic numbers), then asks the engine to fit a linear regression that
predicts one column from the others -- entirely through the setup_data
dict, without touching any algorithm code.

To try a different algorithm, change setup_data["algorithm"]["name"] to
one of: "algorithm" (just loads/returns data), "cluster",
"logistic-regression", "feature-selection". Each one reads a different
subset of setup_data -- see README.md for the full picture, and alg.py for
every key that setup_data can contain.
"""
import os
import tempfile

import pandas as pd

from analytics_engine.algorithms import alg
from analytics_engine.engine import AnalyticEngine


def make_demo_csv(path: str) -> None:
    """A tiny synthetic dataset, standing in for daily site-traffic data."""
    demo_data = pd.DataFrame({
        "day": range(1, 31),
        "qr_scans": [12, 15, 9, 20, 25, 30, 28, 22, 18, 14,
                     16, 19, 23, 27, 31, 29, 24, 20, 17, 15,
                     13, 18, 22, 26, 30, 33, 29, 25, 21, 19],
        "page_views": [40, 48, 33, 61, 74, 88, 82, 67, 55, 44,
                        49, 58, 69, 80, 91, 86, 72, 61, 52, 46,
                        41, 55, 66, 77, 88, 96, 85, 74, 63, 57],
    })
    demo_data.to_csv(path, index=False)


def run_linear_regression_example():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "demo_traffic.csv")
        make_demo_csv(csv_path)

        setup_data = {
            alg.algorithm: {
                alg.name: alg.LinearRegression,
                alg.function: alg.predict,
                # LinearRegressionAlgorithm.predict() takes no arguments,
                # so "input" can be omitted -- shown here for completeness.
                alg.function_input: {},
            },
            alg.data: {
                alg.file: {
                    alg.file_name: os.path.basename(csv_path),
                    alg.path: os.path.dirname(csv_path),
                },
            },
            alg.features: {
                # predict page_views from the other columns.
                alg.target: ["page_views"],
            },
        }

        result = AnalyticEngine(setup_data).handle()
        print("Linear regression R^2 score on the held-out test split:")
        print(result["score"])
        print("\nPredicted vs. held-out rows:")
        print(result["y_predict"])


if __name__ == "__main__":
    run_linear_regression_example()

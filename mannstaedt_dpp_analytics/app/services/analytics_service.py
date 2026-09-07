import pandas as pd


def calculate_qr_share(
    total_views: int,
    qr_views: int,
):
    if total_views == 0:
        return 0

    return round(
        qr_views
        / total_views
        * 100,
        2,
    )


def aggregate_dpp_stats(
    dpp_results: list,
):
    """
    Aggregate company DPP pageview analytics.

    Visitors and visits are intentionally not summed
    because the same person could visit multiple DPPs.
    """

    total_views = sum(
        item[
            "views"
        ]
        for item
        in dpp_results
    )

    total_qr_views = sum(
        item[
            "qr_views"
        ]
        for item
        in dpp_results
    )

    number_of_dpps = len(
        dpp_results
    )

    avg_views = (
        total_views
        / number_of_dpps
        if number_of_dpps > 0
        else 0
    )

    qr_share = calculate_qr_share(
        total_views,
        total_qr_views,
    )

    top_dpp = None

    if dpp_results:

        top_dpp = max(
            dpp_results,
            key=lambda x: x[
                "views"
            ],
        )

    return {
        "tracked_dpps":
            number_of_dpps,

        "total_views":
            total_views,

        "qr_views":
            total_qr_views,

        "qr_share_percent":
            qr_share,

        "avg_views_per_dpp":
            round(
                avg_views,
                1,
            ),

        "top_dpp":
            top_dpp,
    }

def aggregate_metric_lists(metric_lists):
    """
    Combine Umami metric results from several DPPs.

    Example:
    [
        [{"x": "desktop", "y": 10}],
        [{"x": "desktop", "y": 5}, {"x": "mobile", "y": 3}]
    ]

    becomes:

    [
        {"x": "desktop", "y": 15},
        {"x": "mobile", "y": 3}
    ]
    """

    totals = {}

    for metrics in metric_lists:

        for item in metrics:

            name = item.get("x")

            value = item.get(
                "y",
                0,
            )

            if name is None:
                continue

            totals[name] = (
                totals.get(
                    name,
                    0,
                )
                + value
            )

    result = [
        {
            "x": key,
            "y": value,
        }
        for key, value
        in totals.items()
    ]

    result = sorted(
        result,
        key=lambda item: item["y"],
        reverse=True,
    )

    return result


def calculate_non_qr_pageviews(
    total_pageviews,
    qr_pageviews,
):

    return max(
        0,
        total_pageviews
        - qr_pageviews,
    )


def aggregate_timeseries(
    series_list: list,
):
    """
    Combine daily pageview series from multiple DPPs.
    """

    frames = []

    for series in series_list:

        points = series.get(
            "pageviews",
            [],
        )

        if not points:
            continue

        df = pd.DataFrame(
            points
        )

        df["date"] = (
            pd.to_datetime(
                df["x"],
                utc=True,
            )
        )

        df["views"] = (
            pd.to_numeric(
                df["y"],
                errors="coerce",
            )
            .fillna(0)
        )

        frames.append(
            df[
                [
                    "date",
                    "views",
                ]
            ]
        )

    if not frames:

        return pd.DataFrame(
            columns=[
                "date",
                "views",
            ]
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = (
        combined.groupby(
            "date",
            as_index=False,
        )[
            "views"
        ]
        .sum()
        .sort_values(
            "date"
        )
    )

    return combined
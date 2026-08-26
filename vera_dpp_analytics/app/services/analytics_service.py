def calculate_overview(stats: dict):
    visits = stats.get("visits", 0)
    bounces = stats.get("bounces", 0)
    total_time = stats.get("totaltime", 0)

    bounce_rate = (
        (bounces / visits) * 100
        if visits > 0 else 0
    )

    avg_visit_duration = (
        total_time / visits
        if visits > 0 else 0
    )

    return {
        "pageviews": stats.get("pageviews", 0),
        "visitors": stats.get("visitors", 0),
        "visits": visits,
        "bounces": bounces,
        "bounce_rate_percent": round(bounce_rate, 2),
        "avg_visit_duration_seconds": round(avg_visit_duration, 2),
    }


def normalize_metrics(metrics: list):
    """
    Converts:
    [{"x": "DE", "y": 52}]

    into:
    [{"name": "DE", "count": 52}]
    """

    return [
        {
            "name": item.get("x"),
            "count": item.get("y", 0)
        }
        for item in metrics
    ]


def get_top_items(metrics: list, limit: int = 5):
    sorted_items = sorted(
        metrics,
        key=lambda item: item.get("y", 0),
        reverse=True
    )

    return sorted_items[:limit]


def get_least_items(metrics: list, limit: int = 5):
    sorted_items = sorted(
        metrics,
        key=lambda item: item.get("y", 0)
    )

    return sorted_items[:limit]


def calculate_change(current: int, previous: int):
    if previous == 0:
        return None

    return round(
        ((current - previous) / previous) * 100,
        2
    )


def calculate_overview_with_comparison(stats: dict):
    current = calculate_overview(stats)

    comparison_raw = stats.get("comparison", {})

    comparison = calculate_overview(
        comparison_raw
    )

    return {
        "current": current,
        "previous_period": comparison,

        "changes_percent": {
            "pageviews": calculate_change(
                current["pageviews"],
                comparison["pageviews"]
            ),

            "visitors": calculate_change(
                current["visitors"],
                comparison["visitors"]
            ),

            "visits": calculate_change(
                current["visits"],
                comparison["visits"]
            )
        }
    }
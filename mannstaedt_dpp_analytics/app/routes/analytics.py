from fastapi import APIRouter, HTTPException

from app.services.umami_service import (
    get_stats,
    get_pageviews,
    get_metrics,
    get_realtime,
    get_session,
)

from app.services.analytics_service import (
    calculate_overview_with_comparison,
    normalize_metrics,
)

router = APIRouter()


@router.get("/overview")
def overview(start_at: int, end_at: int):
    try:
        stats = get_stats(start_at, end_at)

        return calculate_overview_with_comparison(stats)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/traffic")
def traffic(
    start_at: int,
    end_at: int,
    unit: str = "day"
):
    try:
        return get_pageviews(
            start_at,
            end_at,
            unit=unit
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/countries")
def countries(
    start_at: int,
    end_at: int,
    limit: int = 10
):
    try:
        data = get_metrics(
            start_at,
            end_at,
            metric_type="country",
            limit=limit
        )

        return normalize_metrics(data)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/paths")
def paths(
    start_at: int,
    end_at: int,
    limit: int = 10
):
    try:
        data = get_metrics(
            start_at,
            end_at,
            metric_type="path",
            limit=limit
        )

        return normalize_metrics(data)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/browsers")
def browsers(
    start_at: int,
    end_at: int,
    limit: int = 10
):
    try:
        data = get_metrics(
            start_at,
            end_at,
            metric_type="browser",
            limit=limit
        )

        return normalize_metrics(data)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/referrers")
def referrers(
    start_at: int,
    end_at: int,
    limit: int = 10
):
    try:
        data = get_metrics(
            start_at,
            end_at,
            metric_type="referrer",
            limit=limit
        )

        return normalize_metrics(data)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/realtime")
def realtime():
    try:
        return get_realtime()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/session/{session_id}")
def session(session_id: str):
    try:
        return get_session(session_id)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
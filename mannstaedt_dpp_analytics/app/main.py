from fastapi import FastAPI

from app.routes.analytics import router as analytics_router


app = FastAPI(
    title="DPP Analytics API",
    description="Analytics service for Vera DPP data using Umami.",
    version="0.1.0"
)


app.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["Analytics"]
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "DPP Analytics API"
    }
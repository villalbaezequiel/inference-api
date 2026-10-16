from fastapi import FastAPI

from app.api.routes import health


def create_app() -> FastAPI:
    app = FastAPI(title="AI Inference API", version="0.1.0")
    app.include_router(health.router)
    return app

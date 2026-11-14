"""FastAPI application factory and middleware configuration."""

from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.gateway.router import router as api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Dehallucination Add-On Module Gateway",
        description="Dual-mode API gateway and verification sidecar for hallucination mitigation.",
        version="1.0.0",
    )

    # Enable CORS for open integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["System"])
    async def health_check() -> Dict[str, Any]:
        """System health and version check endpoint."""
        return {"status": "healthy", "version": "1.0.0"}

    # Include gateway and verification routes
    app.include_router(api_router)

    return app

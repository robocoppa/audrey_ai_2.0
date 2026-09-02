"""Native Audrey application resources."""

from fastapi import APIRouter

from audrey.routes.app.me import router as me_router

router = APIRouter(prefix="/api")
router.include_router(me_router)

__all__ = ["router"]

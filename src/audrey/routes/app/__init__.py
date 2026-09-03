"""Native Audrey application resources."""

from fastapi import APIRouter

from audrey.routes.app.conversations import router as conversations_router
from audrey.routes.app.me import router as me_router

router = APIRouter(prefix="/api")
router.include_router(me_router)
router.include_router(conversations_router)

__all__ = ["router"]

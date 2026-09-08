"""Native Audrey application resources."""

from fastapi import APIRouter

from audrey.routes.app.conversations import router as conversations_router
from audrey.routes.app.files import router as files_router
from audrey.routes.app.me import router as me_router
from audrey.routes.app.runs import router as runs_router

router = APIRouter(prefix="/api")
router.include_router(me_router)
router.include_router(conversations_router)
router.include_router(runs_router)
router.include_router(files_router)

__all__ = ["router"]

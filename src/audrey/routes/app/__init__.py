"""Native Audrey application resources."""

from fastapi import APIRouter

from audrey.routes.app.admin import router as admin_router
from audrey.routes.app.agents import router as agents_router
from audrey.routes.app.capabilities import router as capabilities_router
from audrey.routes.app.conversations import router as conversations_router
from audrey.routes.app.files import router as files_router
from audrey.routes.app.knowledge import router as knowledge_router
from audrey.routes.app.me import router as me_router
from audrey.routes.app.models import router as models_router
from audrey.routes.app.private_files import router as private_files_router
from audrey.routes.app.projects import router as projects_router
from audrey.routes.app.runs import router as runs_router
from audrey.routes.app.search import router as search_router
from audrey.routes.app.tax_review import router as tax_review_router

router = APIRouter(prefix="/api")
router.include_router(me_router)
router.include_router(capabilities_router)
router.include_router(models_router)
router.include_router(agents_router)
router.include_router(conversations_router)
router.include_router(projects_router)
router.include_router(knowledge_router)
router.include_router(runs_router)
router.include_router(search_router)
router.include_router(files_router)
router.include_router(private_files_router)
router.include_router(tax_review_router)
router.include_router(admin_router)

__all__ = ["router"]

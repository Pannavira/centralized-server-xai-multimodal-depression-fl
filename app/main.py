import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.model_service import model_service
from app.api.health_routes import router as health_router
from app.api.model_routes import router as model_router
from app.api.screening_routes import router as screening_router

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FED_MIND_SERVER")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load multimodal model onto CUDA/CPU
    logger.info("Initializing FED-MIND Centralized Server...")
    success = model_service.load_model()
    if not success:
        logger.warning("Model could not be loaded at startup! Check checkpoint file.")
    yield
    # Shutdown
    logger.info("Shutting down FED-MIND Centralized Server...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Centralized Application Server for Multimodal Depression Screening and Clinical Explainable AI",
    lifespan=lifespan,
)

# CORS Middleware (mengizinkan koneksi dari klien mobile / dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health_router)
app.include_router(model_router)
app.include_router(screening_router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
    }

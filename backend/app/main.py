from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.core.config import settings
from app.api.routes import search, predict, videos, analytics, virality, classify, trends, topics
from app.api.deps import load_video_cache
from app.ml.embeddings import EmbeddingService
from app.ml.search import SearchService
from app.ml.predictor import EngagementPredictor
from app.ml.virality import ViralityPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ML services on startup."""
    logger.info("=" * 60)
    logger.info("Starting ShofoLens v2.0 - SPECTACULAR Edition")
    logger.info("=" * 60)

    # Load video data cache
    load_video_cache()

    # Initialize embedding service
    logger.info("Loading embedding model...")
    app.state.embedding_service = EmbeddingService()
    logger.info("Embedding service initialized")

    # Initialize search service
    logger.info("Connecting to Qdrant...")
    app.state.search_service = SearchService()
    logger.info("Search service initialized")

    # Initialize basic predictor
    logger.info("Loading engagement predictor...")
    app.state.predictor = EngagementPredictor()
    logger.info("Engagement predictor initialized")

    # Initialize advanced virality predictor
    logger.info("Loading virality predictor...")
    app.state.virality_predictor = ViralityPredictor()
    logger.info("Virality predictor initialized")

    logger.info("=" * 60)
    logger.info("ShofoLens v2.0 API ready!")
    logger.info("Features: Search, Predict, Classify, Trends, Topics, Recommendations")
    logger.info("API docs: http://localhost:8000/docs")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Multimodal Video Intelligence Platform for TikTok Data",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix=settings.API_V1_PREFIX, tags=["Search"])
app.include_router(predict.router, prefix=settings.API_V1_PREFIX, tags=["Prediction"])
app.include_router(virality.router, prefix=settings.API_V1_PREFIX, tags=["Virality"])
app.include_router(classify.router, prefix=settings.API_V1_PREFIX, tags=["Classification"])
app.include_router(trends.router, prefix=settings.API_V1_PREFIX, tags=["Trends"])
app.include_router(topics.router, prefix=settings.API_V1_PREFIX, tags=["Topics"])
app.include_router(videos.router, prefix=settings.API_V1_PREFIX, tags=["Videos"])
app.include_router(analytics.router, prefix=settings.API_V1_PREFIX, tags=["Analytics"])


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "description": "Multimodal Video Intelligence Platform",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}

"""
Content Classification API Endpoints

Zero-shot content classification for automatic labeling.
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

router = APIRouter()


class ClassifyInput(BaseModel):
    description: Optional[str] = Field(None, description="Video description")
    transcript: Optional[str] = Field(None, description="Video transcript")
    hashtags: Optional[List[str]] = Field(None, description="List of hashtags")


class ContentLabel(BaseModel):
    category: str
    confidence: float
    subcategory: Optional[str] = None


class ClassificationResponse(BaseModel):
    primary_category: str
    primary_confidence: float
    all_labels: List[ContentLabel]
    content_type: str
    mood: str


@router.post("/classify", response_model=ClassificationResponse)
async def classify_content(request: Request, video: ClassifyInput):
    """
    Classify video content into categories.

    Uses zero-shot classification to automatically label videos
    without any training data.

    Categories include:
    - Comedy & Entertainment
    - Dance & Music
    - Tutorial & How-To
    - Product Review & Unboxing
    - Food & Cooking
    - And more...
    """
    from app.ml.classifier import get_classifier

    classifier = get_classifier()

    video_data = {
        "description": video.description or "",
        "transcript": video.transcript or "",
        "hashtags": video.hashtags or [],
    }

    result = classifier.classify(video_data)

    return ClassificationResponse(
        primary_category=result.primary_category,
        primary_confidence=result.primary_confidence,
        all_labels=[
            ContentLabel(
                category=label.category,
                confidence=label.confidence,
                subcategory=label.subcategory,
            )
            for label in result.all_labels
        ],
        content_type=result.content_type,
        mood=result.mood,
    )


@router.get("/classify/{video_id}")
async def classify_video_by_id(request: Request, video_id: int):
    """Classify a specific video by its ID."""
    from app.api.routes.videos import get_video_cache
    from app.ml.classifier import get_classifier

    video_cache = get_video_cache()

    if video_id not in video_cache:
        raise HTTPException(status_code=404, detail="Video not found")

    classifier = get_classifier()
    video = video_cache[video_id]
    result = classifier.classify(video)

    return {
        "video_id": video_id,
        "classification": {
            "primary_category": result.primary_category,
            "primary_confidence": result.primary_confidence,
            "all_labels": [
                {"category": l.category, "confidence": l.confidence}
                for l in result.all_labels
            ],
            "content_type": result.content_type,
            "mood": result.mood,
        },
    }


@router.get("/classify/categories/list")
async def list_categories():
    """List all available content categories."""
    from app.ml.classifier import ContentClassifier

    return {
        "categories": ContentClassifier.CATEGORIES,
        "content_types": ContentClassifier.CONTENT_TYPES,
        "moods": ContentClassifier.MOODS,
    }


@router.get("/classify/distribution")
async def get_category_distribution(request: Request):
    """Get distribution of categories across all videos."""
    from app.api.routes.videos import get_video_cache
    from app.ml.classifier import get_classifier, ContentClassifier

    video_cache = get_video_cache()
    videos = list(video_cache.values())

    if not videos:
        return {"distribution": {}, "total": 0}

    classifier = get_classifier()

    # Sample for efficiency
    sample_size = min(500, len(videos))
    import random
    sample = random.sample(videos, sample_size)

    distribution = {cat: 0 for cat in ContentClassifier.CATEGORIES}

    for video in sample:
        result = classifier.classify(video)
        distribution[result.primary_category] += 1

    # Sort by count
    sorted_dist = dict(sorted(
        distribution.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    return {
        "distribution": sorted_dist,
        "sample_size": sample_size,
        "total_videos": len(videos),
    }

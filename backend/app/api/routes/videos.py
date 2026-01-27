from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json

router = APIRouter()


class VideoMetadata(BaseModel):
    video_id: int
    web_url: Optional[str]
    creator: Optional[str]
    description: Optional[str]
    transcript: Optional[str]
    hashtags: List[str]
    duration_ms: Optional[int]
    resolution: Optional[str]
    fps: Optional[float]
    is_ai_generated: bool
    is_ad: bool
    language: Optional[str]
    engagement: Dict[str, int]


class VideoListResponse(BaseModel):
    videos: List[VideoMetadata]
    total: int
    offset: int
    limit: int


# In-memory cache for loaded videos (in production, use Redis/PostgreSQL)
_video_cache: Dict[int, Dict[str, Any]] = {}


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON field that might be a string or already parsed."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return value
    return value


def _format_video(video: Dict[str, Any]) -> VideoMetadata:
    """Format a raw video record into the response model."""
    engagement = _parse_json_field(video.get("engagement_metrics", {})) or {}
    hashtags = _parse_json_field(video.get("hashtags", [])) or []
    language_info = _parse_json_field(video.get("language", {})) or {}

    return VideoMetadata(
        video_id=video.get("video_id", 0),
        web_url=video.get("web_url"),
        creator=video.get("creator"),
        description=video.get("description"),
        transcript=video.get("transcript", "")[:500] if video.get("transcript") else None,
        hashtags=hashtags if isinstance(hashtags, list) else [],
        duration_ms=video.get("duration_ms"),
        resolution=video.get("resolution"),
        fps=video.get("fps"),
        is_ai_generated=bool(video.get("is_ai_generated")),
        is_ad=bool(video.get("is_ad")),
        language=language_info.get("desc_language") if isinstance(language_info, dict) else None,
        engagement={
            "play_count": engagement.get("play_count", 0) or 0,
            "like_count": engagement.get("like_count", 0) or 0,
            "share_count": engagement.get("share_count", 0) or 0,
            "comment_count": engagement.get("comment_count", 0) or 0,
        },
    )


@router.get("/videos", response_model=VideoListResponse)
async def list_videos(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    sort_by: str = Query("play_count", description="Sort field"),
    order: str = Query("desc", description="Sort order (asc/desc)"),
):
    """
    List videos with pagination and sorting.
    """
    # Get videos from cache
    videos = list(_video_cache.values())

    # Sort
    if sort_by in ["play_count", "like_count", "share_count"]:
        def get_sort_key(v):
            engagement = _parse_json_field(v.get("engagement_metrics", {})) or {}
            return engagement.get(sort_by, 0) or 0

        videos.sort(key=get_sort_key, reverse=(order == "desc"))
    elif sort_by == "duration_ms":
        videos.sort(key=lambda v: v.get("duration_ms", 0) or 0, reverse=(order == "desc"))

    # Paginate
    total = len(videos)
    videos = videos[offset:offset + limit]

    return VideoListResponse(
        videos=[_format_video(v) for v in videos],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/videos/{video_id}", response_model=VideoMetadata)
async def get_video(request: Request, video_id: int):
    """
    Get a specific video by ID.
    """
    if video_id not in _video_cache:
        raise HTTPException(status_code=404, detail="Video not found")

    return _format_video(_video_cache[video_id])


@router.get("/videos/{video_id}/comments")
async def get_video_comments(request: Request, video_id: int, limit: int = 50):
    """
    Get comments for a specific video.
    """
    if video_id not in _video_cache:
        raise HTTPException(status_code=404, detail="Video not found")

    video = _video_cache[video_id]
    comments = _parse_json_field(video.get("comments", [])) or []

    if isinstance(comments, list):
        comments = comments[:limit]
    else:
        comments = []

    return {
        "video_id": video_id,
        "comments": comments,
        "total": len(comments),
    }


@router.get("/videos/random/sample")
async def get_random_sample(request: Request, count: int = Query(10, ge=1, le=50)):
    """
    Get a random sample of videos.
    """
    import random

    videos = list(_video_cache.values())
    if len(videos) == 0:
        return {"videos": [], "total": 0}

    sample = random.sample(videos, min(count, len(videos)))

    return {
        "videos": [_format_video(v) for v in sample],
        "total": len(sample),
    }


def load_videos_to_cache(videos: List[Dict[str, Any]]):
    """Load videos into the in-memory cache."""
    global _video_cache
    _video_cache = {v.get("video_id", i): v for i, v in enumerate(videos)}


def get_video_cache() -> Dict[int, Dict[str, Any]]:
    """Get the video cache."""
    return _video_cache


@router.get("/videos/{video_id}/similar")
async def get_similar_videos(
    request: Request,
    video_id: int,
    limit: int = Query(10, ge=1, le=50),
    method: str = Query("hybrid", description="Recommendation method: content, engagement, hybrid"),
):
    """
    Get similar video recommendations.

    Args:
        video_id: Source video ID
        limit: Number of recommendations
        method: Recommendation method
            - content: Based on content similarity
            - engagement: Based on engagement patterns
            - hybrid: Combined approach (recommended)

    Returns:
        List of similar videos with similarity scores and explanations.
    """
    from app.ml.recommendations import get_recommendation_engine

    if video_id not in _video_cache:
        raise HTTPException(status_code=404, detail="Video not found")

    engine = get_recommendation_engine()

    # Index videos if not already done
    if not engine.video_metadata:
        engine.index_videos(list(_video_cache.values()))

    result = engine.get_similar(video_id, n=limit, method=method)

    return {
        "source_video_id": result.source_video_id,
        "recommendation_type": result.recommendation_type,
        "recommendations": [
            {
                "video_id": rec.video_id,
                "similarity_score": rec.similarity_score,
                "explanation": rec.explanation,
                "shared_hashtags": rec.shared_hashtags,
                "engagement_similarity": rec.engagement_similarity,
                "video": _format_video(_video_cache[rec.video_id]) if rec.video_id in _video_cache else None,
            }
            for rec in result.recommendations
        ],
    }


@router.get("/videos/{video_id}/analyze")
async def analyze_video(request: Request, video_id: int):
    """
    Get comprehensive analysis for a video.

    Combines classification, virality prediction, and recommendations.
    """
    from app.ml.classifier import get_classifier

    if video_id not in _video_cache:
        raise HTTPException(status_code=404, detail="Video not found")

    video = _video_cache[video_id]
    classifier = get_classifier()

    # Classify content
    classification = classifier.classify(video)

    # Get similar videos (quick version)
    from app.ml.recommendations import get_recommendation_engine
    engine = get_recommendation_engine()
    if not engine.video_metadata:
        engine.index_videos(list(_video_cache.values()))

    similar = engine.get_similar(video_id, n=5, method="hybrid")

    return {
        "video": _format_video(video),
        "classification": {
            "primary_category": classification.primary_category,
            "confidence": classification.primary_confidence,
            "content_type": classification.content_type,
            "mood": classification.mood,
        },
        "similar_videos": [
            {
                "video_id": rec.video_id,
                "similarity_score": rec.similarity_score,
                "explanation": rec.explanation,
            }
            for rec in similar.recommendations[:5]
        ],
    }

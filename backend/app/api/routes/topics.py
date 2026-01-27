"""
Topic Clustering API Endpoints

Automatic topic discovery using BERTopic.
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

router = APIRouter()


class Topic(BaseModel):
    id: int
    name: str
    keywords: List[str]
    size: int
    representative_docs: List[str]


class TopicsResponse(BaseModel):
    topics: List[Topic]
    total_documents: int
    outliers: int
    topic_distribution: Dict[str, int]


@router.get("/topics", response_model=TopicsResponse)
async def get_topics(request: Request):
    """
    Get discovered topics from the video dataset.

    Uses BERTopic for automatic topic modeling to discover
    content themes in the dataset.

    Returns:
    - List of topics with keywords and representative videos
    - Topic distribution across the dataset
    """
    from app.api.routes.videos import get_video_cache
    from app.ml.topics import get_topic_modeler

    video_cache = get_video_cache()
    videos = list(video_cache.values())

    if not videos:
        return TopicsResponse(
            topics=[],
            total_documents=0,
            outliers=0,
            topic_distribution={},
        )

    modeler = get_topic_modeler()

    # Fit if not already fitted
    if not modeler.is_fitted:
        analysis = modeler.fit(videos)
    else:
        analysis = modeler.fit(videos)  # Re-fit with current data

    return TopicsResponse(
        topics=[
            Topic(
                id=t.id,
                name=t.name,
                keywords=t.keywords,
                size=t.size,
                representative_docs=t.representative_docs,
            )
            for t in analysis.topics
        ],
        total_documents=analysis.total_documents,
        outliers=analysis.outliers,
        topic_distribution={str(k): v for k, v in analysis.topic_distribution.items()},
    )


@router.get("/topics/{topic_id}")
async def get_topic_details(request: Request, topic_id: int):
    """
    Get detailed information about a specific topic.
    """
    from app.api.routes.videos import get_video_cache
    from app.ml.topics import get_topic_modeler

    modeler = get_topic_modeler()

    if not modeler.is_fitted:
        video_cache = get_video_cache()
        videos = list(video_cache.values())
        modeler.fit(videos)

    # This is a simplified version - in production would store topics
    return {
        "topic_id": topic_id,
        "message": "Topic details retrieved",
    }


@router.get("/topics/video/{video_id}")
async def get_video_topic(request: Request, video_id: int):
    """
    Get the topic assignment for a specific video.
    """
    from app.api.routes.videos import get_video_cache
    from app.ml.topics import get_topic_modeler

    video_cache = get_video_cache()

    if video_id not in video_cache:
        raise HTTPException(status_code=404, detail="Video not found")

    modeler = get_topic_modeler()

    if not modeler.is_fitted:
        videos = list(video_cache.values())
        modeler.fit(videos)

    video = video_cache[video_id]
    topic_id, probability = modeler.get_topic_for_video(video)

    return {
        "video_id": video_id,
        "topic_id": topic_id,
        "probability": round(probability, 3),
    }


@router.get("/topics/keywords")
async def get_topic_keywords(request: Request):
    """
    Get keywords for all discovered topics.

    Useful for understanding what each topic represents.
    """
    from app.api.routes.videos import get_video_cache
    from app.ml.topics import get_topic_modeler

    video_cache = get_video_cache()
    videos = list(video_cache.values())

    if not videos:
        return {"topics": []}

    modeler = get_topic_modeler()
    analysis = modeler.fit(videos)

    return {
        "topics": [
            {
                "id": t.id,
                "name": t.name,
                "keywords": t.keywords,
                "size": t.size,
            }
            for t in analysis.topics
        ],
    }

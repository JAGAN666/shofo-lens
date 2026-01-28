from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

router = APIRouter()


class SearchQuery(BaseModel):
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(20, ge=1, le=100, description="Number of results to return")
    min_views: Optional[int] = Field(None, description="Minimum view count filter")
    language: Optional[str] = Field(None, description="Filter by language code")
    has_transcript: Optional[bool] = Field(None, description="Filter to videos with transcripts")


class SearchResult(BaseModel):
    video_id: int
    score: float
    description: Optional[str]
    transcript: Optional[str]
    creator: Optional[str]
    web_url: Optional[str]
    play_count: Optional[int]
    like_count: Optional[int]
    duration_ms: Optional[int]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int


def _fallback_search(query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Fallback keyword-based search when Qdrant has no data."""
    from app.api.routes.videos import get_video_cache
    import json

    video_cache = get_video_cache()
    if not video_cache:
        return []

    query_words = query_text.lower().split()
    scored_videos = []

    for video_id, video in video_cache.items():
        score = 0.0

        # Check description
        description = (video.get("description") or "").lower()
        for word in query_words:
            if word in description:
                score += 0.3

        # Check transcript
        transcript = (video.get("transcript") or "").lower()
        for word in query_words:
            if word in transcript:
                score += 0.4

        # Check hashtags
        hashtags = video.get("hashtags", [])
        if isinstance(hashtags, str):
            try:
                hashtags = json.loads(hashtags)
            except:
                hashtags = []
        hashtags_text = " ".join(hashtags).lower()
        for word in query_words:
            if word in hashtags_text or f"#{word}" in hashtags_text:
                score += 0.3

        if score > 0:
            engagement = video.get("engagement_metrics", {})
            if isinstance(engagement, str):
                try:
                    engagement = json.loads(engagement)
                except:
                    engagement = {}

            scored_videos.append({
                "video_id": video_id,
                "score": min(score, 0.99),
                "description": video.get("description"),
                "transcript": video.get("transcript"),
                "creator": video.get("creator"),
                "web_url": video.get("web_url"),
                "play_count": engagement.get("play_count"),
                "like_count": engagement.get("like_count"),
                "duration_ms": video.get("duration_ms"),
            })

    # Sort by score descending
    scored_videos.sort(key=lambda x: x["score"], reverse=True)
    return scored_videos[:limit]


@router.post("/search", response_model=SearchResponse)
async def semantic_search(request: Request, query: SearchQuery):
    """
    Perform semantic search across TikTok videos.

    Searches using natural language queries like:
    - "cooking videos with hand gestures"
    - "funny dance transitions"
    - "product reviews and unboxing"
    """
    embedding_service = request.app.state.embedding_service
    search_service = request.app.state.search_service

    # Generate embedding for the query
    query_embedding = embedding_service.encode_single(query.query)

    # Build filters
    filters = {}
    if query.min_views:
        filters["min_views"] = query.min_views
    if query.language:
        filters["language"] = query.language
    if query.has_transcript is not None:
        filters["has_transcript"] = query.has_transcript

    # Perform search
    results = search_service.search(
        query_embedding=query_embedding,
        limit=query.limit,
        filters=filters if filters else None,
    )

    # Fallback to keyword search if Qdrant returns no results
    if not results:
        results = _fallback_search(query.query, query.limit)

    return SearchResponse(
        query=query.query,
        results=[
            SearchResult(
                video_id=r["video_id"],
                score=r["score"],
                description=r.get("description"),
                transcript=r.get("transcript", "")[:500] if r.get("transcript") else None,
                creator=r.get("creator"),
                web_url=r.get("web_url"),
                play_count=r.get("play_count"),
                like_count=r.get("like_count"),
                duration_ms=r.get("duration_ms"),
            )
            for r in results
        ],
        total=len(results),
    )


@router.get("/search/stats")
async def get_search_stats(request: Request):
    """Get statistics about the search index."""
    search_service = request.app.state.search_service
    return search_service.get_collection_stats()

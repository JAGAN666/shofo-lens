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

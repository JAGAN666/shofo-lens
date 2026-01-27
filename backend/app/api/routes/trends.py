"""
Trend Detection API Endpoints

Real-time trend analysis for hashtags, topics, and content.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any

router = APIRouter()


class TrendItem(BaseModel):
    name: str
    count: int
    velocity: float
    avg_engagement: float
    trend_status: str
    rank_change: int


class TrendResponse(BaseModel):
    trending_hashtags: List[TrendItem]
    trending_topics: List[str]
    emerging_hashtags: List[TrendItem]
    declining_hashtags: List[TrendItem]
    best_posting_times: List[Dict[str, Any]]
    engagement_insights: Dict[str, Any]


@router.get("/trends", response_model=TrendResponse)
async def get_trends(request: Request):
    """
    Get comprehensive trend analysis.

    Returns:
    - Trending hashtags with velocity and engagement metrics
    - Trending topics extracted from content
    - Emerging hashtags (rising fast)
    - Declining hashtags (losing momentum)
    - Best posting times based on engagement
    - Overall engagement insights
    """
    from app.api.routes.videos import get_video_cache
    from app.ml.trends import get_trend_detector

    video_cache = get_video_cache()
    videos = list(video_cache.values())

    if not videos:
        return TrendResponse(
            trending_hashtags=[],
            trending_topics=[],
            emerging_hashtags=[],
            declining_hashtags=[],
            best_posting_times=[],
            engagement_insights={},
        )

    detector = get_trend_detector()
    report = detector.analyze(videos)

    return TrendResponse(
        trending_hashtags=[
            TrendItem(
                name=t.name,
                count=t.count,
                velocity=t.velocity,
                avg_engagement=t.avg_engagement,
                trend_status=t.trend_status,
                rank_change=t.rank_change,
            )
            for t in report.trending_hashtags
        ],
        trending_topics=report.trending_topics,
        emerging_hashtags=[
            TrendItem(
                name=t.name,
                count=t.count,
                velocity=t.velocity,
                avg_engagement=t.avg_engagement,
                trend_status=t.trend_status,
                rank_change=t.rank_change,
            )
            for t in report.emerging_hashtags
        ],
        declining_hashtags=[
            TrendItem(
                name=t.name,
                count=t.count,
                velocity=t.velocity,
                avg_engagement=t.avg_engagement,
                trend_status=t.trend_status,
                rank_change=t.rank_change,
            )
            for t in report.declining_hashtags
        ],
        best_posting_times=report.best_posting_times,
        engagement_insights=report.engagement_insights,
    )


@router.get("/trends/hashtags")
async def get_trending_hashtags(
    request: Request,
    limit: int = 20,
    status: str = None,
):
    """
    Get trending hashtags with optional filtering.

    Args:
        limit: Number of hashtags to return
        status: Filter by trend status (Rising, Hot, Stable, Declining)
    """
    from app.api.routes.videos import get_video_cache
    from app.ml.trends import get_trend_detector

    video_cache = get_video_cache()
    videos = list(video_cache.values())

    if not videos:
        return {"hashtags": [], "total": 0}

    detector = get_trend_detector()
    report = detector.analyze(videos)

    hashtags = report.trending_hashtags

    if status:
        hashtags = [h for h in hashtags if h.trend_status.lower() == status.lower()]

    return {
        "hashtags": [
            {
                "name": h.name,
                "count": h.count,
                "velocity": h.velocity,
                "avg_engagement": h.avg_engagement,
                "trend_status": h.trend_status,
                "rank_change": h.rank_change,
            }
            for h in hashtags[:limit]
        ],
        "total": len(hashtags),
    }


@router.get("/trends/posting-times")
async def get_best_posting_times(request: Request):
    """
    Get best posting times based on engagement analysis.

    Returns hourly breakdown of engagement rates to help
    determine optimal posting schedule.
    """
    from app.api.routes.videos import get_video_cache
    from app.ml.trends import get_trend_detector

    video_cache = get_video_cache()
    videos = list(video_cache.values())

    if not videos:
        return {"posting_times": [], "recommendation": ""}

    detector = get_trend_detector()
    report = detector.analyze(videos)

    best_times = report.best_posting_times[:5]

    recommendation = ""
    if best_times:
        top_hours = [t["formatted"] for t in best_times[:3]]
        recommendation = f"Best times to post: {', '.join(top_hours)}"

    return {
        "posting_times": report.best_posting_times,
        "recommendation": recommendation,
        "total_videos_analyzed": len(videos),
    }


@router.get("/trends/emerging")
async def get_emerging_trends(request: Request, limit: int = 10):
    """
    Get emerging trends (hashtags/topics gaining traction fast).

    These are trends with high velocity but not yet mainstream.
    """
    from app.api.routes.videos import get_video_cache
    from app.ml.trends import get_trend_detector

    video_cache = get_video_cache()
    videos = list(video_cache.values())

    if not videos:
        return {"emerging": [], "total": 0}

    detector = get_trend_detector()
    report = detector.analyze(videos)

    return {
        "emerging": [
            {
                "name": h.name,
                "count": h.count,
                "velocity": h.velocity,
                "avg_engagement": h.avg_engagement,
                "potential": "High" if h.velocity > 0.5 else "Medium",
            }
            for h in report.emerging_hashtags[:limit]
        ],
        "total": len(report.emerging_hashtags),
    }

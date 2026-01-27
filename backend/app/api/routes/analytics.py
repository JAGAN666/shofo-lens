from fastapi import APIRouter, Request
from typing import Dict, Any, List
from collections import Counter
import json

router = APIRouter()


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


@router.get("/analytics/overview")
async def get_overview(request: Request):
    """
    Get an overview of the dataset analytics.
    """
    from app.api.routes.videos import get_video_cache

    videos = list(get_video_cache().values())

    if not videos:
        return {
            "total_videos": 0,
            "total_views": 0,
            "total_likes": 0,
            "avg_duration_sec": 0,
            "videos_with_transcript": 0,
            "ai_generated_videos": 0,
            "ad_videos": 0,
        }

    total_views = 0
    total_likes = 0
    total_duration = 0
    with_transcript = 0
    ai_generated = 0
    ads = 0

    for video in videos:
        engagement = _parse_json_field(video.get("engagement_metrics", {})) or {}
        total_views += engagement.get("play_count", 0) or 0
        total_likes += engagement.get("like_count", 0) or 0
        total_duration += video.get("duration_ms", 0) or 0

        if video.get("transcript"):
            with_transcript += 1
        if video.get("is_ai_generated"):
            ai_generated += 1
        if video.get("is_ad"):
            ads += 1

    return {
        "total_videos": len(videos),
        "total_views": total_views,
        "total_likes": total_likes,
        "avg_duration_sec": round(total_duration / len(videos) / 1000, 2) if videos else 0,
        "videos_with_transcript": with_transcript,
        "transcript_percentage": round(with_transcript / len(videos) * 100, 1) if videos else 0,
        "ai_generated_videos": ai_generated,
        "ad_videos": ads,
    }


@router.get("/analytics/top-hashtags")
async def get_top_hashtags(request: Request, limit: int = 20):
    """
    Get the most used hashtags in the dataset.
    """
    from app.api.routes.videos import get_video_cache

    videos = list(get_video_cache().values())
    hashtag_counter: Counter = Counter()

    for video in videos:
        hashtags = _parse_json_field(video.get("hashtags", [])) or []
        if isinstance(hashtags, list):
            hashtag_counter.update(hashtags)

    top_hashtags = hashtag_counter.most_common(limit)

    return {
        "hashtags": [
            {"tag": tag, "count": count}
            for tag, count in top_hashtags
        ],
        "total_unique": len(hashtag_counter),
    }


@router.get("/analytics/engagement-distribution")
async def get_engagement_distribution(request: Request):
    """
    Get the distribution of engagement metrics.
    """
    from app.api.routes.videos import get_video_cache

    videos = list(get_video_cache().values())

    views = []
    likes = []
    engagement_rates = []

    for video in videos:
        engagement = _parse_json_field(video.get("engagement_metrics", {})) or {}
        play_count = engagement.get("play_count", 0) or 0
        like_count = engagement.get("like_count", 0) or 0

        views.append(play_count)
        likes.append(like_count)

        if play_count > 0:
            engagement_rates.append(like_count / play_count)

    def calculate_percentiles(data: List[float]) -> Dict[str, float]:
        if not data:
            return {"p25": 0, "p50": 0, "p75": 0, "p90": 0, "p99": 0}

        sorted_data = sorted(data)
        n = len(sorted_data)

        return {
            "p25": sorted_data[int(n * 0.25)],
            "p50": sorted_data[int(n * 0.50)],
            "p75": sorted_data[int(n * 0.75)],
            "p90": sorted_data[int(n * 0.90)],
            "p99": sorted_data[int(n * 0.99)] if n > 100 else sorted_data[-1],
        }

    return {
        "views": {
            "min": min(views) if views else 0,
            "max": max(views) if views else 0,
            "mean": sum(views) / len(views) if views else 0,
            "percentiles": calculate_percentiles(views),
        },
        "likes": {
            "min": min(likes) if likes else 0,
            "max": max(likes) if likes else 0,
            "mean": sum(likes) / len(likes) if likes else 0,
            "percentiles": calculate_percentiles(likes),
        },
        "engagement_rate": {
            "min": min(engagement_rates) if engagement_rates else 0,
            "max": max(engagement_rates) if engagement_rates else 0,
            "mean": sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0,
            "percentiles": calculate_percentiles(engagement_rates),
        },
    }


@router.get("/analytics/language-distribution")
async def get_language_distribution(request: Request, limit: int = 15):
    """
    Get the distribution of languages in the dataset.
    """
    from app.api.routes.videos import get_video_cache

    videos = list(get_video_cache().values())
    language_counter: Counter = Counter()

    for video in videos:
        language_info = _parse_json_field(video.get("language", {})) or {}
        if isinstance(language_info, dict):
            lang = language_info.get("desc_language")
            if lang:
                language_counter[lang] += 1

    top_languages = language_counter.most_common(limit)

    return {
        "languages": [
            {"code": lang, "count": count, "percentage": round(count / len(videos) * 100, 1)}
            for lang, count in top_languages
        ],
        "total_languages": len(language_counter),
    }


@router.get("/analytics/duration-distribution")
async def get_duration_distribution(request: Request):
    """
    Get the distribution of video durations.
    """
    from app.api.routes.videos import get_video_cache

    videos = list(get_video_cache().values())

    # Bucket durations
    buckets = {
        "0-15s": 0,
        "15-30s": 0,
        "30-60s": 0,
        "1-3min": 0,
        "3min+": 0,
    }

    for video in videos:
        duration_ms = video.get("duration_ms", 0) or 0
        duration_sec = duration_ms / 1000

        if duration_sec <= 15:
            buckets["0-15s"] += 1
        elif duration_sec <= 30:
            buckets["15-30s"] += 1
        elif duration_sec <= 60:
            buckets["30-60s"] += 1
        elif duration_sec <= 180:
            buckets["1-3min"] += 1
        else:
            buckets["3min+"] += 1

    return {
        "distribution": [
            {"bucket": bucket, "count": count, "percentage": round(count / len(videos) * 100, 1) if videos else 0}
            for bucket, count in buckets.items()
        ],
        "total_videos": len(videos),
    }


@router.get("/analytics/creators")
async def get_top_creators(request: Request, limit: int = 20):
    """
    Get the most prolific creators in the dataset.
    """
    from app.api.routes.videos import get_video_cache

    videos = list(get_video_cache().values())
    creator_stats: Dict[str, Dict[str, int]] = {}

    for video in videos:
        creator = video.get("creator")
        if not creator:
            continue

        if creator not in creator_stats:
            creator_stats[creator] = {"videos": 0, "total_views": 0, "total_likes": 0}

        engagement = _parse_json_field(video.get("engagement_metrics", {})) or {}
        creator_stats[creator]["videos"] += 1
        creator_stats[creator]["total_views"] += engagement.get("play_count", 0) or 0
        creator_stats[creator]["total_likes"] += engagement.get("like_count", 0) or 0

    # Sort by video count
    sorted_creators = sorted(
        creator_stats.items(),
        key=lambda x: x[1]["videos"],
        reverse=True
    )[:limit]

    return {
        "creators": [
            {
                "username": creator,
                "video_count": stats["videos"],
                "total_views": stats["total_views"],
                "total_likes": stats["total_likes"],
                "avg_views": round(stats["total_views"] / stats["videos"]) if stats["videos"] else 0,
            }
            for creator, stats in sorted_creators
        ],
        "total_creators": len(creator_stats),
    }

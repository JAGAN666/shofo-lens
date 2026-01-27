"""
Trend Detection Algorithm

Identifies emerging hashtags, content types, and topics based on:
- Velocity of growth (how fast something is gaining traction)
- Volume (total count)
- Engagement rate correlation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np
import json
from loguru import logger


@dataclass
class TrendItem:
    """A single trending item (hashtag, topic, etc.)."""
    name: str
    count: int
    velocity: float  # Growth rate
    avg_engagement: float
    trend_status: str  # "Rising", "Hot", "Stable", "Declining"
    rank_change: int  # Position change from previous period


@dataclass
class TrendReport:
    """Complete trend analysis report."""
    trending_hashtags: List[TrendItem]
    trending_topics: List[str]
    emerging_hashtags: List[TrendItem]  # New/fast-growing
    declining_hashtags: List[TrendItem]  # Losing momentum
    best_posting_times: List[Dict[str, Any]]
    engagement_insights: Dict[str, Any]


class TrendDetector:
    """
    Detects trends in video content based on hashtags, engagement,
    and temporal patterns.
    """

    def __init__(self):
        self.hashtag_history: Dict[str, List[int]] = defaultdict(list)
        self.previous_rankings: Dict[str, int] = {}

    def analyze(self, videos: List[Dict[str, Any]]) -> TrendReport:
        """
        Perform comprehensive trend analysis on video dataset.

        Args:
            videos: List of video data dictionaries

        Returns:
            TrendReport with all trend insights
        """
        # Extract and analyze hashtags
        hashtag_data = self._analyze_hashtags(videos)
        trending = self._identify_trending(hashtag_data)
        emerging = self._identify_emerging(hashtag_data)
        declining = self._identify_declining(hashtag_data)

        # Analyze posting times
        best_times = self._analyze_posting_times(videos)

        # Extract topics from hashtags
        topics = self._extract_topics(hashtag_data)

        # Engagement insights
        engagement_insights = self._analyze_engagement_patterns(videos)

        return TrendReport(
            trending_hashtags=trending[:15],
            trending_topics=topics[:10],
            emerging_hashtags=emerging[:10],
            declining_hashtags=declining[:10],
            best_posting_times=best_times,
            engagement_insights=engagement_insights,
        )

    def _analyze_hashtags(self, videos: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze hashtag usage and engagement."""
        hashtag_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_views": 0, "total_likes": 0, "videos": []}
        )

        for video in videos:
            hashtags = self._parse_json(video.get("hashtags", []))
            if not isinstance(hashtags, list):
                continue

            engagement = self._parse_json(video.get("engagement_metrics", {})) or {}
            views = engagement.get("play_count", 0) or 0
            likes = engagement.get("like_count", 0) or 0

            for tag in hashtags[:20]:  # Limit to first 20 hashtags
                tag = str(tag).lower().strip()
                if len(tag) < 2:
                    continue

                hashtag_stats[tag]["count"] += 1
                hashtag_stats[tag]["total_views"] += views
                hashtag_stats[tag]["total_likes"] += likes
                hashtag_stats[tag]["videos"].append(video)

        # Calculate averages and velocity
        for tag, stats in hashtag_stats.items():
            if stats["count"] > 0:
                stats["avg_views"] = stats["total_views"] / stats["count"]
                stats["avg_likes"] = stats["total_likes"] / stats["count"]
                stats["engagement_rate"] = (
                    stats["total_likes"] / stats["total_views"]
                    if stats["total_views"] > 0
                    else 0
                )

                # Calculate velocity based on recent videos
                recent_count = sum(
                    1 for v in stats["videos"][-100:]  # Look at last 100 videos
                )
                old_count = max(self.hashtag_history.get(tag, [0])[-1:], [0])[0]
                stats["velocity"] = (recent_count - old_count) / max(old_count, 1)

                # Update history
                self.hashtag_history[tag].append(stats["count"])

        return hashtag_stats

    def _identify_trending(self, hashtag_data: Dict[str, Dict[str, Any]]) -> List[TrendItem]:
        """Identify currently trending hashtags."""
        trending = []

        for tag, stats in hashtag_data.items():
            if stats["count"] < 5:  # Minimum threshold
                continue

            # Score based on count, engagement, and velocity
            score = (
                np.log1p(stats["count"]) * 0.3
                + np.log1p(stats["avg_views"]) * 0.3
                + stats["engagement_rate"] * 1000 * 0.2
                + max(stats["velocity"], 0) * 0.2
            )

            # Determine trend status
            velocity = stats["velocity"]
            if velocity > 0.5:
                status = "Rising"
            elif velocity > 0.1:
                status = "Hot"
            elif velocity > -0.1:
                status = "Stable"
            else:
                status = "Declining"

            # Calculate rank change
            prev_rank = self.previous_rankings.get(tag, 999)
            current_rank = len(trending)
            rank_change = prev_rank - current_rank

            trending.append(TrendItem(
                name=tag,
                count=stats["count"],
                velocity=round(velocity, 2),
                avg_engagement=round(stats["engagement_rate"] * 100, 2),
                trend_status=status,
                rank_change=rank_change,
            ))

        # Sort by score (higher is better)
        trending.sort(key=lambda x: x.count * (1 + x.velocity), reverse=True)

        # Update rankings for next comparison
        self.previous_rankings = {t.name: i for i, t in enumerate(trending)}

        return trending

    def _identify_emerging(self, hashtag_data: Dict[str, Dict[str, Any]]) -> List[TrendItem]:
        """Identify newly emerging hashtags with high velocity."""
        emerging = []

        for tag, stats in hashtag_data.items():
            # Look for tags with moderate count but high velocity
            if 3 <= stats["count"] <= 500 and stats["velocity"] > 0.3:
                emerging.append(TrendItem(
                    name=tag,
                    count=stats["count"],
                    velocity=round(stats["velocity"], 2),
                    avg_engagement=round(stats["engagement_rate"] * 100, 2),
                    trend_status="Emerging",
                    rank_change=0,
                ))

        # Sort by velocity
        emerging.sort(key=lambda x: x.velocity, reverse=True)
        return emerging

    def _identify_declining(self, hashtag_data: Dict[str, Dict[str, Any]]) -> List[TrendItem]:
        """Identify hashtags losing momentum."""
        declining = []

        for tag, stats in hashtag_data.items():
            if stats["count"] > 10 and stats["velocity"] < -0.2:
                declining.append(TrendItem(
                    name=tag,
                    count=stats["count"],
                    velocity=round(stats["velocity"], 2),
                    avg_engagement=round(stats["engagement_rate"] * 100, 2),
                    trend_status="Declining",
                    rank_change=0,
                ))

        # Sort by velocity (most declining first)
        declining.sort(key=lambda x: x.velocity)
        return declining

    def _analyze_posting_times(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze best posting times based on engagement."""
        time_stats: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_engagement": 0}
        )

        for video in videos:
            date_posted = video.get("date_posted")
            if not date_posted:
                continue

            try:
                if isinstance(date_posted, str):
                    dt = datetime.fromisoformat(date_posted.replace("Z", "+00:00"))
                elif isinstance(date_posted, (int, float)):
                    dt = datetime.fromtimestamp(date_posted)
                else:
                    continue

                hour = dt.hour
                engagement = self._parse_json(video.get("engagement_metrics", {})) or {}
                likes = engagement.get("like_count", 0) or 0
                views = engagement.get("play_count", 0) or 0

                eng_rate = likes / views if views > 0 else 0

                time_stats[hour]["count"] += 1
                time_stats[hour]["total_engagement"] += eng_rate

            except:
                continue

        # Calculate average engagement per hour
        best_times = []
        for hour, stats in time_stats.items():
            if stats["count"] > 0:
                avg_engagement = stats["total_engagement"] / stats["count"]
                best_times.append({
                    "hour": hour,
                    "formatted": f"{hour:02d}:00",
                    "video_count": stats["count"],
                    "avg_engagement": round(avg_engagement * 100, 2),
                })

        # Sort by engagement
        best_times.sort(key=lambda x: x["avg_engagement"], reverse=True)
        return best_times[:10]

    def _extract_topics(self, hashtag_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """Extract high-level topics from hashtags."""
        # Group related hashtags into topics
        topic_keywords = {
            "Entertainment": ["funny", "comedy", "meme", "viral", "fyp"],
            "Music & Dance": ["dance", "music", "song", "dj", "remix"],
            "Food & Lifestyle": ["food", "recipe", "cooking", "vlog", "life"],
            "Beauty & Fashion": ["beauty", "makeup", "fashion", "style", "ootd"],
            "Fitness": ["fitness", "workout", "gym", "health", "motivation"],
            "Gaming": ["gaming", "game", "gamer", "streamer"],
            "Education": ["learn", "tips", "howto", "tutorial", "hack"],
            "Pets": ["pet", "dog", "cat", "animal", "cute"],
        }

        topic_scores: Dict[str, float] = defaultdict(float)

        for tag, stats in hashtag_data.items():
            for topic, keywords in topic_keywords.items():
                if any(kw in tag.lower() for kw in keywords):
                    topic_scores[topic] += stats["count"] * (1 + stats["engagement_rate"])

        # Sort topics by score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics]

    def _analyze_engagement_patterns(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall engagement patterns."""
        total_views = 0
        total_likes = 0
        total_shares = 0
        total_comments = 0
        count = 0

        engagement_rates = []

        for video in videos:
            engagement = self._parse_json(video.get("engagement_metrics", {})) or {}
            views = engagement.get("play_count", 0) or 0
            likes = engagement.get("like_count", 0) or 0
            shares = engagement.get("share_count", 0) or 0
            comments = engagement.get("comment_count", 0) or 0

            total_views += views
            total_likes += likes
            total_shares += shares
            total_comments += comments
            count += 1

            if views > 0:
                engagement_rates.append(likes / views)

        avg_engagement = np.mean(engagement_rates) if engagement_rates else 0
        median_engagement = np.median(engagement_rates) if engagement_rates else 0

        return {
            "total_videos": count,
            "total_views": total_views,
            "total_likes": total_likes,
            "total_shares": total_shares,
            "total_comments": total_comments,
            "avg_engagement_rate": round(avg_engagement * 100, 2),
            "median_engagement_rate": round(median_engagement * 100, 2),
            "avg_views_per_video": round(total_views / count) if count > 0 else 0,
            "avg_likes_per_video": round(total_likes / count) if count > 0 else 0,
        }

    def _parse_json(self, value: Any) -> Any:
        """Parse JSON field."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return value
        return value


# Singleton instance
_trend_detector: Optional[TrendDetector] = None


def get_trend_detector() -> TrendDetector:
    """Get or create trend detector instance."""
    global _trend_detector
    if _trend_detector is None:
        _trend_detector = TrendDetector()
    return _trend_detector
